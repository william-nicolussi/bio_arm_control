import os
import time
import mujoco
import numpy as np

from numba import njit
from src.lut.lut import Lut
from src.lut.cost_function import u_from_y, cost_total, cost_derivative

# No cache=True: numba's cache does not see changes in cost_function.py, and would keep the old cost

@njit
def _u(lam, K, W, p):
    """The 6 equations of the notes with lambda fixed: u_i = u_from_y(lam K_i) for every muscle."""
    u = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        u[i] = u_from_y(lam * K[i], W, p)
    return u

@njit
def _residual(lam, tau_0, K, tau_ref, W, p):
    return tau_0 + np.dot(K, _u(lam, K, W, p)) - tau_ref

@njit
def _solve_interior(tau_0, K, tau_ref, W, p, tol):
    """Zero of the residual in lambda. The residual is increasing in lambda for every
    separable, strictly convex cost: nothing here depends on the form of J."""
    # bracket: find lam_lo with residual < 0 and lam_hi with residual > 0
    lam_lo = -1.0
    r_lo = _residual(lam_lo, tau_0, K, tau_ref, W, p)
    while r_lo > 0: # torque still too high: go left
        lam_lo *= 2
        r_lo = _residual(lam_lo, tau_0, K, tau_ref, W, p)
    lam_hi = 1.0
    r_hi = _residual(lam_hi, tau_0, K, tau_ref, W, p)
    while r_hi < 0: # torque still too low: go right
        lam_hi *= 2
        r_hi = _residual(lam_hi, tau_0, K, tau_ref, W, p)

    # Illinois method: secant between the two ends of the bracket, the zero always stays inside.
    # Converges much faster than bisection; when the same end moves twice in a row, the value
    # at the other end is halved, so that end moves too (plain regula falsi can get stuck)
    lam = 0.5 * (lam_lo + lam_hi) # numba wants lam defined before the loop
    side = 0                      # which end moved last: -1 lam_lo, +1 lam_hi
    for _ in range(200):
        lam = (lam_lo * r_hi - lam_hi * r_lo) / (r_hi - r_lo) # zero of the line through the two ends
        r = _residual(lam, tau_0, K, tau_ref, W, p)
        if abs(r) < tol: # torque error below tol Nm: solved
            break
        if r < 0: # torque too low: the solution is to the right
            lam_lo, r_lo = lam, r
            if side == -1:
                r_hi *= 0.5
            side = -1
        else:     # torque too high: the solution is to the left
            lam_hi, r_hi = lam, r
            if side == 1:
                r_lo *= 0.5
            side = 1
    return _u(lam, K, W, p), lam

@njit
def _tau_min_max(tau_0, K):
    """Torque range of one state: muscles with K_i < 0 at 1 (tau_min) or with K_i > 0 at 1 (tau_max).
    The only place where tau_min and tau_max are computed: the grid and the edge test below
    must give the same bits, otherwise tau_ref = tau_min could end up in _solve_interior."""
    tau_min = tau_0
    tau_max = tau_0
    for k in K:
        if k < 0:
            tau_min += k
        else:
            tau_max += k
    return tau_min, tau_max

@njit
def _solve_point(tau_0, K, tau_ref, W, p, tol):
    """The 7 equations of the notes at one point: u (6,) and lam."""
    tau_min, tau_max = _tau_min_max(tau_0, K)
    # edges: only one u gives tau_min (or tau_max), and lambda would be -inf (or +inf)
    if tau_ref <= tau_min:
        return (K < 0).astype(np.float64), -np.inf
    if tau_ref >= tau_max:
        return (K > 0).astype(np.float64), np.inf
    return _solve_interior(tau_0, K, tau_ref, W, p, tol)

@njit
def _solve_angle(tau_0, K, tau, W, p, tol):
    """All the points of one angle, with no Python in between: K (V, M), tau (V, T) -> u (V, T, M), lam (V, T)."""
    V, T = tau.shape
    M = K.shape[1]
    u = np.zeros((V, T, M))
    lam = np.zeros((V, T))
    for v in range(V):
        for t in range(T):
            u_vt, lam_vt = _solve_point(tau_0, K[v], tau[v, t], W, p, tol)
            u[v, t, :] = u_vt
            lam[v, t] = lam_vt
    return u, lam

class LutGenerator:
    def __init__(self, path_model: str, p_value: float, w_values):

        # parameter of lut
        self.w_values = np.atleast_1d(np.asarray(w_values, dtype=float))
        if np.any((self.w_values < 0) | (self.w_values > 1)):
            raise ValueError("W values must be in [0, 1]")
        if p_value <= 1:
            raise ValueError("p must be > 1: the cost must be strictly convex")
        self.path_model = path_model
        self.p_value = float(p_value)

        # MuJoCo model, used to read tau_0 and K
        self._model = mujoco.MjModel.from_xml_path(path_model)
        self._data = mujoco.MjData(self._model)
        if self._model.nv != 1 or self._model.na != self._model.nu:
            raise ValueError("expected 1 DoF and one activation state per muscle")
        self.muscles = [self._model.actuator(i).name for i in range(self._model.nu)]

        # state grids
        self.theta = None
        self.theta_dot = None
        self.tau_norm = None

        # muscle maps: tau = tau_0 + sum K_i u_i
        self.tau_0 = None
        self.K = None

        self.tau_min = None
        self.tau_max = None
        self.tau = None
        
        # solution of the Lagrangian: unknowns u_i and lambda
        self.u = None
        self.lam = None
        
        # useful
        self.lut = None
        self.nc_path = None
  
    # ----- MAIN METHOD -----
    def generate_lut(self, lut_name: str, torque_points: int, angle_points: int,
                     max_theta_dot: float, velocity_points: int, edge_density: float) -> None:
        
        if np.ndim(velocity_points) == 0 and velocity_points % 2 == 0:
            raise ValueError("velocity_points must be odd, so that 0 is a grid node")
        if max_theta_dot <= 0:
            raise ValueError("max_theta_dot must be > 0")
        if not 0 <= edge_density < 1:
            raise ValueError("edge_density must be in [0, 1)")
        
        start = time.perf_counter()

        self.populate_theta(angle_points)
        self.populate_theta_dot(max_theta_dot, velocity_points)
        self.populate_tau_norm(torque_points, edge_density)

        self.populate_tau_0()
        self.populate_K()
        self.populate_tau_min_max()
        self.populate_tau()

        self.populate_u_lam()

        err = self.check_constraint()
        print(f"LUT generated in {time.perf_counter() - start:.1f} s: "
              f"max |tau_0 + K u - tau_ref| = {err:.1e} Nm")

        self.lut = Lut(lut_name=lut_name,
                       model_name=os.path.basename(self.path_model),
                       p=self.p_value,
                       muscles_names=self.muscles,
                       edge_density=edge_density,
                       w=self.w_values,
                       theta=self.theta,
                       theta_dot=self.theta_dot,
                       tau=self.tau,
                       tau_0=self.tau_0,
                       k=self.K,
                       lam=self.lam,
                       tau_norm=self.tau_norm,
                       tau_min=self.tau_min,
                       tau_max=self.tau_max,
                       activations=self.u)

    # ---- To perform calculations ----
    def _torque(self, theta: float, theta_dot: float, act: np.ndarray) -> float:
        """Muscle torque [Nm] at one state (deg, deg/s) with activations act: tau(u) of the notes.
        qfrc_actuator contains only the muscles: no gravity, no joint damping."""
        m = self._model
        d = self._data
        d.qpos[0] = np.deg2rad(theta) # MuJoCo works in rad and rad/s
        d.qvel[0] = np.deg2rad(theta_dot)
        d.act[:] = act
        mujoco.mj_forward(m, d) # computes the state, does not advance in time
        torque = float(d.qfrc_actuator[0])
        return torque

    def get_tau0(self, theta: float) -> float:
        """tau_0(theta) = sum r_i F0_i F_P(L_i): torque with all activations at 0.
        F_P depends only on the muscle length, so only on the angle: theta_dot = 0."""
        tau_0 = self._torque(theta, 0.0, np.zeros(self._model.nu))
        return tau_0

    def get_K(self, theta: float, theta_dot: float) -> np.ndarray:
        """K_i(theta, theta_dot) = r_i F0_i F_L(L_i) F_V(V_i) for every muscle.
        Exact because tau is linear in act: K_i = tau(only muscle i at 1) - tau(all at 0)."""
        n_muscles = self._model.nu
        tau_rest = self._torque(theta, theta_dot, np.zeros(n_muscles))   # = tau_0(theta)

        K = np.zeros(n_muscles)
        for i in range(n_muscles):
            act = np.zeros(n_muscles)
            act[i] = 1.0
            K[i] = self._torque(theta, theta_dot, act) - tau_rest
        return K
    
    def get_tau(self, tau_0: float, K: np.ndarray, u: np.ndarray) -> float:
        """tau(u) = tau_0 + sum K_i u_i: the linear model of the notes. No MuJoCo here."""
        return tau_0 + np.dot(K, u)

    def get_u_min(self, K: np.ndarray) -> np.ndarray:
        """Activations that give the minimum torque: muscles with K_i < 0 at 1, the others at 0."""
        return (K < 0).astype(float)

    def get_u_max(self, K: np.ndarray) -> np.ndarray:
        """Activations that give the maximum torque: muscles with K_i > 0 at 1, the others at 0."""
        return (K > 0).astype(float)

    def get_tau_min(self, tau_0: float, K: np.ndarray) -> float:
        return _tau_min_max(float(tau_0), np.ascontiguousarray(K, dtype=np.float64))[0]

    def get_tau_max(self, tau_0: float, K: np.ndarray) -> float:
        return _tau_min_max(float(tau_0), np.ascontiguousarray(K, dtype=np.float64))[1]

    def get_tau_from_norm(self, tau_norm, tau_min, tau_max):
        """tau_norm in [0, 1] -> tau [Nm]. Used when building the LUT: tau_norm is the grid we choose."""
        return (1.0 - tau_norm) * tau_min + tau_norm * tau_max   # exact at tau_norm = 0 and 1

    # ---- Populate parameters ----
    def populate_theta(self, angles) -> None:
        """angles: number of uniform points on the joint range of the model, or the angles themselves [deg]
        (increasing, inside the joint range: e.g. the nodes of grid_design.nodes_minimax)."""
        m = self._model
        if m.njnt != 1 or not m.jnt_limited[0]:
            raise ValueError("expected a model with one limited joint")
        theta_min, theta_max = np.rad2deg(m.jnt_range[0])   # [0] = the only joint; XML is in rad

        if np.ndim(angles) == 0:                             # a number: uniform grid, as before
            self.theta = np.linspace(theta_min, theta_max, int(angles))
            return
        theta = np.asarray(angles, dtype=float)
        if np.any(np.diff(theta) <= 0):
            raise ValueError("the angles must be strictly increasing")
        if theta[0] < theta_min - 1e-9 or theta[-1] > theta_max + 1e-9:
            raise ValueError(f"the angles must be inside the joint range [{theta_min:g}, {theta_max:g}] deg")
        self.theta = theta
        
    def populate_theta_dot(self, max_theta_dot, velocities) -> None:
        """velocities: number of uniform points on [-max_theta_dot, max_theta_dot], or the velocities themselves
        [deg/s] (increasing, inside the range: e.g. the midpoints of a training grid, for validation)."""
        if np.ndim(velocities) == 0:
            self.theta_dot = np.linspace(-max_theta_dot, max_theta_dot, int(velocities))
            return
        theta_dot = np.asarray(velocities, dtype=float)
        if np.any(np.diff(theta_dot) <= 0):
            raise ValueError("the velocities must be strictly increasing")
        if np.any(np.abs(theta_dot) > max_theta_dot + 1e-9):
            raise ValueError(f"the velocities must be inside [-{max_theta_dot:g}, {max_theta_dot:g}] deg/s")
        self.theta_dot = theta_dot

    def populate_tau_norm(self, torques, edge_density) -> None:
        """torques: number of points in [0, 1] moved towards 0 and 1 by edge_density (0 = uniform), or the
        torque_norm values themselves (increasing, in [0, 1]); edge_density is then not used."""
        if np.ndim(torques) == 0:
            s = np.linspace(0.0, 1.0, int(torques))                 # uniform grid
            self.tau_norm = s - edge_density * np.sin(2 * np.pi * s) / (2 * np.pi)
            return
        tau_norm = np.asarray(torques, dtype=float)
        if np.any(np.diff(tau_norm) <= 0):
            raise ValueError("the torque_norm values must be strictly increasing")
        if tau_norm[0] < 0.0 or tau_norm[-1] > 1.0:
            raise ValueError("the torque_norm values must be in [0, 1]")
        self.tau_norm = tau_norm

    def populate_tau_0(self):
        self.tau_0 = np.zeros(len(self.theta))
        for a, theta in enumerate(self.theta):
            self.tau_0[a] = self.get_tau0(theta)

    def populate_K(self) -> None:
        """self.K (A, V, M): K[a, v, i] = K_i at theta[a], theta_dot[v].
        These are the 6 maps (theta, theta_dot) -> K_i of the notes: map i is self.K[:, :, i]."""
        A, V, M = len(self.theta), len(self.theta_dot), self._model.nu
        self.K = np.zeros((A, V, M))
        for a, theta in enumerate(self.theta):
            for v, theta_dot in enumerate(self.theta_dot):
                self.K[a, v] = self.get_K(theta, theta_dot)    # vector of 6 values

    def populate_tau_min_max(self) -> None:
        """self.tau_min and self.tau_max (A, V): torque range of every state (theta, theta_dot)."""
        A, V = len(self.theta), len(self.theta_dot)
        self.tau_min = np.zeros((A, V))
        self.tau_max = np.zeros((A, V))
        for a in range(A):
            for v in range(V):
                self.tau_min[a, v] = self.get_tau_min(self.tau_0[a], self.K[a, v])
                self.tau_max[a, v] = self.get_tau_max(self.tau_0[a], self.K[a, v])

    def populate_tau(self) -> None:
        """self.tau (A, V, T): the tau_ref of the notes, T values for every state,
        all inside [tau_min, tau_max] of that state."""
        # broadcasting: tau_norm (1, 1, T), tau_min and tau_max (A, V, 1) -> (A, V, T)
        self.tau = self.get_tau_from_norm(self.tau_norm[np.newaxis, np.newaxis, :],
                                          self.tau_min[:, :, np.newaxis],
                                          self.tau_max[:, :, np.newaxis])

    def populate_u_lam(self) -> None:
        """self.u (W, A, V, T, M) and self.lam (W, A, V, T): solution of the system at every point."""
        nW = len(self.w_values)
        A = len(self.theta)
        V, T, M = len(self.theta_dot), len(self.tau_norm), self._model.nu
        self.u = np.zeros((nW, A, V, T, M))
        self.lam = np.zeros((nW, A, V, T))
        for w, W in enumerate(self.w_values):
            start_W = time.perf_counter()
            for a in range(A):
                print(f"\r  W = {W:g}: angle {a + 1}/{A}", end="", flush=True)   # progress
                # all (velocity, torque) of this angle in one compiled call
                self.u[w, a], self.lam[w, a] = _solve_angle(
                    float(self.tau_0[a]), self.K[a], self.tau[a], float(W), self.p_value, 1e-10)
            print(f"\nW = {W:g} done in {time.perf_counter() - start_W:.1f} s")    
    
    def check_constraint(self, tol: float = 1e-6) -> float:
        """The 7th equation must hold at every point: |tau_0 + sum K_i u_i - tau_ref| < tol.
        Returns the largest error [Nm]; raises if it is above tol."""
        # tau_0 + sum_i K_i u_i at every point: K (A, V, M), u (nW, A, V, T, M) -> (nW, A, V, T)
        tau_check = (self.tau_0[np.newaxis, :, np.newaxis, np.newaxis]
                     + np.einsum("avm,wavtm->wavt", self.K, self.u))
        err = float(np.max(np.abs(tau_check - self.tau[np.newaxis])))
        if err > tol:
            raise RuntimeError(f"torque constraint not satisfied: max error {err:.2e} Nm")
        return err

    # ---- Solve the system ----
    def solve_system(self, tau_0: float, K: np.ndarray, tau_ref: float, W: float,
                     tol: float = 1e-10):
        """Solves the 7 equations of the notes at one point:
            f'(u_i) = lam K_i  (6 eq.)    and    tau_0 + sum K_i u_i = tau_ref  (1 eq.)
        Returns u (6,) and lam."""
        # edges and interior, compiled with numba (see _solve_point)
        return _solve_point(float(tau_0), np.ascontiguousarray(K, dtype=np.float64),
                            float(tau_ref), float(W), self.p_value, tol)

    def get_u(self, lam: float, K: np.ndarray, W: float) -> np.ndarray:
        """The 6 equations of the notes with lambda fixed: u_i(lambda) for every muscle."""
        return _u(float(lam), np.ascontiguousarray(K, dtype=np.float64), float(W), self.p_value)

    def get_u_i(self, lam: float, K_i: float, W: float) -> float:
        """ONE of the 6 equations, lambda fixed: J_i'(u_i) = lam K_i, clipped to [0, 1] (see cost_function.u_from_y)."""
        return u_from_y(lam * K_i, float(W), self.p_value)

    # ----- ONLINE ALLOCATION -----
    def get_activations(self, W: float, angle: float, torque: float, velocity: float = 0.0) -> np.ndarray:
        """Activations (1, 6) solving the system at the current state, without the LUT: same interface as
        LUT.get_activations (W, angle [deg], torque [Nm], velocity [deg/s]), so it can be used as an
        allocator in simulate_closed_loop. No need to call generate_lut: tau_0 and K are computed with
        MuJoCo at (angle, velocity), so the force-velocity effect of the muscles is included and the
        angle range is the one of the model. Outside [tau_min, tau_max] the muscles saturate
        (u_min or u_max) instead of returning NaN."""
        tau_0 = self.get_tau0(angle)
        K = self.get_K(angle, velocity)
        u, _ = self.solve_system(tau_0, K, float(torque), float(W))
        return u.reshape(1, -1)

    # ----- INFO AND CHECKS -----
    def check_solver(self, n_tests: int = 50, seed: int = 0) -> float:
        """Compare the solver with scipy SLSQP on random states (call it after generate_lut). Returns the largest
        |u_solver - u_slsqp|; raises if SLSQP finds a lower cost (the solver would be wrong)."""
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        lo, hi = np.rad2deg(self._model.jnt_range[0])
        p = self.p_value
        max_du = 0.0
        for _ in range(n_tests):
            W = float(rng.choice(self.w_values))
            theta = rng.uniform(lo, hi)
            theta_dot = rng.uniform(self.theta_dot[0], self.theta_dot[-1])
            tau_0 = self.get_tau0(theta)
            K = self.get_K(theta, theta_dot)
            t_min = self.get_tau_min(tau_0, K)
            t_max = self.get_tau_max(tau_0, K)
            tau_ref = t_min + rng.uniform(0.02, 0.98) * (t_max - t_min)

            u, _ = self.solve_system(tau_0, K, tau_ref, W)

            def cost(x):
                return cost_total(x, W, p)

            res = minimize(cost, np.full(len(K), 0.5), method="SLSQP",
                           jac=lambda x: cost_derivative(x, W, p),
                           bounds=[(1e-12, 1.0)] * len(K),
                           constraints={"type": "eq", "fun": lambda x: tau_0 + K @ x - tau_ref,
                                        "jac": lambda x: K},
                           options={"ftol": 1e-14, "maxiter": 500})
            if cost(res.x) < cost(u) - 1e-9:
                raise RuntimeError(f"SLSQP found a lower cost at theta={theta:.1f}, "
                                   f"theta_dot={theta_dot:.1f}, W={W:g}: solver wrong")
            max_du = max(max_du, np.abs(u - res.x).max())
        print(f"check_solver: max |u - u_SLSQP| = {max_du:.1e} on {n_tests} random states")
        return max_du