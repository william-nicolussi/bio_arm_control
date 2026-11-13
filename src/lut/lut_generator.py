import os
import time
import math
import mujoco
import numpy as np
import xarray as xr
from numba import njit

@njit(cache=True)
def _u_i(lam, K_i, W, p):
    y = lam * K_i
    a = p * (1 - W)

    if y >= a + W:
        return 1.0
    if W == 0: # W = 0:  p u^(p-1) = y  ->  u = (y/p)^(1/(p-1));  if y <= 0 the muscle is off
        return (y / p) ** (1 / (p - 1)) if y > 0 else 0.0
    if W == 1: # W = 1:  ln u + 1 = y  ->  u = e^(y-1)
        return math.exp(y - 1)

    # 0 < W < 1: no closed form, bisection on s = ln u  (so u = e^s, u^(p-1) = e^((p-1)s))
    s_lo = min((y - W) / W, math.log(W / a) / (p - 1)) - 1.0   # here left side < y
    s_hi = 0.0                                                   # here left side > y (u = 1)
    for _ in range(200):
        s = 0.5 * (s_lo + s_hi)
        left = a * math.exp((p - 1) * s) + W * (s + 1)
        if left < y:
            s_lo = s # u too small: the solution is to the right
        else:
            s_hi = s # u too large: the solution is to the left
        if s_hi - s_lo < 1e-13 * (1.0 + abs(s_lo)): # relative: works for any size of s
            break
    return math.exp(0.5 * (s_lo + s_hi))

@njit(cache=True)
def _u(lam, K, W, p):
    u = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        u[i] = _u_i(lam, K[i], W, p)
    return u

@njit(cache=True)
def _residual(lam, tau_0, K, tau_ref, W, p):
    return tau_0 + np.dot(K, _u(lam, K, W, p)) - tau_ref

@njit(cache=True)
def _solve_interior(tau_0, K, tau_ref, W, p, tol):
    # bracket: find lam_lo with residual < 0 and lam_hi with residual > 0
    lam_lo = -1.0
    while _residual(lam_lo, tau_0, K, tau_ref, W, p) > 0: # torque still too high: go left
        lam_lo *= 2
    lam_hi = 1.0
    while _residual(lam_hi, tau_0, K, tau_ref, W, p) < 0: # torque still too low: go right
        lam_hi *= 2

    # bisection: halve [lam_lo, lam_hi] keeping the zero inside
    lam = 0.5 * (lam_lo + lam_hi) # numba wants lam defined before the loop
    for _ in range(200):
        lam = 0.5 * (lam_lo + lam_hi)
        r = _residual(lam, tau_0, K, tau_ref, W, p)
        if abs(r) < tol: # torque error below tol Nm: solved
            break
        if r < 0:
            lam_lo = lam # torque too low: the solution is to the right
        else:
            lam_hi = lam # torque too high: the solution is to the left
    return _u(lam, K, W, p), lam

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
    def generate_lut(self, torque_points: int, angle_points: int,
                     max_theta_dot: float, velocity_points: int, edge_density: float) -> xr.Dataset:
        
        if velocity_points % 2 == 0:
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

        self.lut = self._to_dataset()
        return self.lut

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
        return self.get_tau(tau_0, K, self.get_u_min(K))

    def get_tau_max(self, tau_0: float, K: np.ndarray) -> float:
        return self.get_tau(tau_0, K, self.get_u_max(K))

    def get_tau_from_norm(self, tau_norm, tau_min, tau_max):
        """tau_norm in [0, 1] -> tau [Nm]. Used when building the LUT: tau_norm is the grid we choose."""
        return (1.0 - tau_norm) * tau_min + tau_norm * tau_max   # exact at tau_norm = 0 and 1

    # ---- Populate parameters ----
    def populate_theta(self, angle_points) -> None:
        m = self._model
        if m.njnt != 1 or not m.jnt_limited[0]:
            raise ValueError("expected a model with one limited joint")
        theta_min, theta_max = np.rad2deg(m.jnt_range[0])   # [0] = the only joint; XML is in rad
        self.theta = np.linspace(theta_min, theta_max, angle_points)

    def populate_theta_dot(self, max_theta_dot, velocity_points):
        self.theta_dot = np.linspace(-max_theta_dot, max_theta_dot, velocity_points)

    def populate_tau_norm(self, torque_points, edge_density):
        """self.tau_norm: torque_points values in [0, 1], moved towards 0 and 1 by edge_density.
        edge_density = 0 -> uniform grid."""
        s = np.linspace(0.0, 1.0, torque_points)                    # uniform grid
        self.tau_norm = s - edge_density * np.sin(2 * np.pi * s) / (2 * np.pi)

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
        A, V, T = len(self.theta), len(self.theta_dot), len(self.tau_norm)
        self.tau = np.zeros((A, V, T))
        for a in range(A):
            for v in range(V):
                for t in range(T):
                    self.tau[a, v, t] = self.get_tau_from_norm(
                        self.tau_norm[t], self.tau_min[a, v], self.tau_max[a, v])

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
                for v in range(V):
                    for t in range(T):
                        self.u[w, a, v, t], self.lam[w, a, v, t] = self.solve_system(
                            self.tau_0[a], self.K[a, v], self.tau[a, v, t], W)
            print(f"\nW = {W:g} done in {time.perf_counter() - start_W:.1f} s")    
    
    def check_constraint(self, tol: float = 1e-6) -> float:
        """The 7th equation must hold at every point: |tau_0 + sum K_i u_i - tau_ref| < tol.
        Returns the largest error [Nm]; raises if it is above tol."""
        err = 0.0
        for w in range(len(self.w_values)):
            for a in range(len(self.theta)):
                for v in range(len(self.theta_dot)):
                    for t in range(len(self.tau_norm)):
                        tau_check = self.get_tau(self.tau_0[a], self.K[a, v], self.u[w, a, v, t])
                        err = max(err, abs(tau_check - self.tau[a, v, t]))
        if err > tol:
            raise RuntimeError(f"torque constraint not satisfied: max error {err:.2e} Nm")
        return err

    # ---- Solve the system ----
    def solve_system(self, tau_0: float, K: np.ndarray, tau_ref: float, W: float,
                     tol: float = 1e-10):
        """Solves the 7 equations of the notes at one point:
            f'(u_i) = lam K_i  (6 eq.)    and    tau_0 + sum K_i u_i = tau_ref  (1 eq.)
        Returns u (6,) and lam."""
        # edges: only one u gives tau_min (or tau_max), and lambda would be -inf (or +inf)
        if tau_ref <= self.get_tau_min(tau_0, K):
            return self.get_u_min(K), -math.inf
        if tau_ref >= self.get_tau_max(tau_0, K):
            return self.get_u_max(K), math.inf

        # interior: bracket + bisection on lambda (compiled with numba, see _solve_interior)
        return _solve_interior(tau_0, K, tau_ref, W, self.p_value, tol)

    def get_u(self, lam: float, K: np.ndarray, W: float) -> np.ndarray:
        """The 6 equations of the notes with lambda fixed: u_i(lambda) for every muscle."""
        return _u(lam, K, W, self.p_value)

    def get_u_i(self, lam: float, K_i: float, W: float) -> float:
        """ONE of the 6 equations, lambda fixed: p(1-W) u_i^(p-1) + W (ln u_i + 1) = lam K_i, clipped to [0, 1]."""
        return _u_i(lam, K_i, W, self.p_value)

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

    # ----- OUTPUT -----
    def _to_dataset(self) -> xr.Dataset:
        """LUT as an xarray Dataset. The names inside the file (angle, velocity, torque_norm,
        torque, ...) are the ones read by lut.py: do not rename them."""
        ds = xr.Dataset(
            data_vars={
                "activation_value": (("W", "angle", "velocity", "torque_norm", "activation_idx"), self.u),
                "torque": (("angle", "velocity", "torque_norm"), self.tau),
                "torque_min": (("angle", "velocity"), self.tau_min),
                "torque_max": (("angle", "velocity"), self.tau_max),
                "tau_0": (("angle",), self.tau_0),
                "K": (("angle", "velocity", "activation_idx"), self.K),
            },
            coords={
                "W": self.w_values,
                "angle": self.theta,             # deg
                "velocity": self.theta_dot,      # deg/s
                "torque_norm": self.tau_norm,
                "activation_idx": np.arange(len(self.muscles)),
                "muscle": ("activation_idx", self.muscles),
            },
        )
        ds.attrs.update(p=self.p_value,
                        model_file=os.path.basename(self.path_model))
        return ds

    def save(self, nc_file_path: str) -> None:
        if self.lut is None:
            raise RuntimeError("LUT not generated yet: call generate_lut() first")
        os.makedirs(os.path.dirname(os.path.abspath(nc_file_path)), exist_ok=True)
        # temporary file + replace: on Windows the target may be open, and a file on disk is always complete
        temp_file = nc_file_path + ".tmp"
        self.lut.to_netcdf(temp_file)
        os.replace(temp_file, nc_file_path)
        self.nc_path = nc_file_path
        print(f"File NC saved in: {nc_file_path}")

    def to_csv(self, csv_file_path: str) -> None:
        """CSV export, delegated to LUT (one implementation only). Needs the LUT saved first."""
        from src.lut import Lut
        Lut(self._saved_path()).to_csv(csv_file_path)

    def save_heatmaps(self, folder: str, velocities=None) -> None:
        """Heatmaps (angle, torque_norm) of every W at some velocities, delegated to LUT.
        velocities: list in deg/s; default: minimum, 0 and maximum of the grid."""
        from src.lut import Lut
        if velocities is None:
            velocities = [self.theta_dot[0], 0.0, self.theta_dot[-1]]
        lut = Lut(self._saved_path())
        lut.load_data()
        os.makedirs(folder, exist_ok=True)
        for W in self.w_values:
            for v in velocities:
                path = os.path.join(folder, f"heatmap_W_{W:g}_vel_{v:g}_p{self.p_value:g}.png")
                lut.plot_heatmaps(considered_W=float(W), save_path=path, show=False, velocity=v)

    def _saved_path(self) -> str:
        if self.nc_path is None:
            raise RuntimeError("LUT not saved yet: call save() first")
        return self.nc_path

    # ----- INFO AND CHECKS -----
    def check_solver(self, n_tests: int = 50, seed: int = 0) -> float:
        """Compare the solver with scipy SLSQP on random states (call it after generate_lut). Returns the largest
        |u_solver - u_slsqp|; raises if SLSQP finds a lower cost (the solver would be wrong)."""
        from scipy.optimize import minimize
        from scipy.special import xlogy   # x ln x with 0 ln 0 = 0

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
                return (1 - W) * np.sum(x ** p) + W * np.sum(xlogy(x, x))

            res = minimize(cost, np.full(len(K), 0.5), method="SLSQP",
                           jac=lambda x: p * (1 - W) * x ** (p - 1) + W * (np.log(np.maximum(x, 1e-300)) + 1),
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