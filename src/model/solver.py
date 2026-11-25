import numpy as np
from numba import njit

from src.lut.cost_function import u_from_y, u_rest


# ----- numba -----
@njit
def _activations(lam, K, W, p, u):
    """u_i = u_from_y(lam^T K_i) for every muscle (in place)."""
    n, M = K.shape
    for i in range(M):
        y = 0.0
        for j in range(n):
            y += lam[j] * K[j, i]
        u[i] = u_from_y(y, W, p)


@njit
def _du_dy(u, W, p):
    """du/dy = 1 / J''(u), 0 where u is clipped (u = 1, or u = 0 when W = 0)."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 1.0 / (p * (p - 1.0) * (1.0 - W) * u ** (p - 2.0) + W / u)


@njit
def _residual(lam, tau_0, K, tau_ref, W, p, u):
    _activations(lam, K, W, p, u)
    return tau_0 + K @ u - tau_ref


@njit
def _slope(lam, step, t, tau_0, K, tau_ref, W, p, u):
    """phi'(t) of phi(t) = D(lam - t step), D the (concave) dual: grad D = -r, so phi'(t) = r(lam - t step)^T step."""
    r = _residual(lam - t * step, tau_0, K, tau_ref, W, p, u)
    return np.dot(r, step)


@njit
def _solve(tau_0, K, tau_ref, W, p, tol, max_iter):
    """Dual ascent: Newton direction + exact line search on the concave dual D(lam).
    Only residuals are needed: dD/dlam = -r. Returns u, lam, iterations, |r|."""
    n, M = K.shape
    lam = np.zeros(n)
    u = np.empty(M)
    r = _residual(lam, tau_0, K, tau_ref, W, p, u)
    scale2 = np.max(np.abs(K)) ** 2 + 1e-24
    it = 0
    for it in range(max_iter):
        if np.sqrt(np.sum(r ** 2)) < tol:
            break
        # Newton direction, with a small Levenberg term: J is singular where every muscle is clipped
        J = np.zeros((n, n))
        for i in range(M):
            d = _du_dy(u[i], W, p)
            for a in range(n):
                for b in range(n):
                    J[a, b] += K[a, i] * K[b, i] * d
        mu = 1e-9 * (np.trace(J) + scale2)
        for a in range(n):
            J[a, a] += mu
        step = np.linalg.solve(J, r)            # lam_new = lam - t step;  phi'(0) = r^T step > 0

        # exact line search: phi'(t) decreases (D concave); bracket its zero, then Illinois
        g0 = np.dot(r, step)
        t_lo, g_lo = 0.0, g0
        t_hi, g_hi = 1.0, _slope(lam, step, 1.0, tau_0, K, tau_ref, W, p, u)
        while g_hi > 0.0 and t_hi < 1e12:       # still going up: go further
            t_lo, g_lo = t_hi, g_hi
            t_hi *= 2.0
            g_hi = _slope(lam, step, t_hi, tau_0, K, tau_ref, W, p, u)
        t = t_hi
        if g_hi < -1e-3 * g0:                    # t_hi overshoots: Illinois on [t_lo, t_hi]
            side = 0
            for _ in range(60):
                t = (t_lo * g_hi - t_hi * g_lo) / (g_hi - g_lo)
                g = _slope(lam, step, t, tau_0, K, tau_ref, W, p, u)
                if abs(g) <= 1e-3 * g0:
                    break
                if g > 0.0:
                    t_lo, g_lo = t, g
                    if side == 1:
                        g_hi *= 0.5
                    side = 1
                else:
                    t_hi, g_hi = t, g
                    if side == -1:
                        g_lo *= 0.5
                    side = -1
        lam = lam - t * step
        r = _residual(lam, tau_0, K, tau_ref, W, p, u)
    return u, lam, it, np.sqrt(np.sum(r ** 2))


@njit
def _boundary_distance(tau_0, K, tau_ref):
    """rho of tau_ref in the zonotope (n = 1 or 2). Facet normals: 1 in 1D, the K_i rotated by 90 deg in 2D."""
    n, M = K.shape
    c = tau_0 + 0.5 * K.sum(axis=1)
    d = tau_ref - c
    rho = 0.0
    if n == 1:
        half = 0.5 * np.sum(np.abs(K[0]))
        return abs(d[0]) / max(half, 1e-12)
    for i in range(M):
        nx, ny = -K[1, i], K[0, i]                  # normal of the facets parallel to K_i
        if nx == 0.0 and ny == 0.0:
            continue
        half = 0.0
        for j in range(M):
            half += abs(nx * K[0, j] + ny * K[1, j])
        half *= 0.5
        if half > 0.0:
            rho = max(rho, abs(nx * d[0] + ny * d[1]) / half)
    return rho


# ----- python -----
def _boundary_activations(tau_0, K, point, W, p, tol, max_iter) -> np.ndarray:
    """u (M,) that gives the torque point, on the boundary of the zonotope (rho = 1), for n = 1 or 2."""
    n, M = K.shape
    d = point - (tau_0 + 0.5 * K.sum(axis=1))
    if n == 1:
        normal = np.sign(d)
    else:   # facet with the largest rho: its normal is a K_i rotated by 90 deg (as in _boundary_distance)
        normals = np.column_stack([-K[1], K[0]])                        # (M, 2)
        half = 0.5 * np.abs(normals @ K).sum(axis=1)                     # support of the zonotope
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(half > 0, np.abs(normals @ d) / half, -np.inf)
        normal = normals[np.argmax(ratio)]
        normal = normal * np.sign(normal @ d)
    push = normal @ K                                                   # > 0: the muscle pushes outwards
    free = np.abs(push) <= 1e-9 * np.abs(normal).max() * np.abs(K).max()
    u = (push > 0).astype(np.float64)
    if not free.any():
        return u
    if n == 1:          # K_i = 0: the muscle does not change the torque, its best value is the rest one
        u[free] = u_rest(W, p)
        return u
    # muscles parallel to the facet: 1D problem along it, with the torque the fixed muscles leave
    along = np.array([-normal[1], normal[0]])
    K_free = np.ascontiguousarray((along @ K[:, free]).reshape(1, -1))
    left = np.array([along @ (point - tau_0 - K[:, ~free] @ u[~free])])
    if _boundary_distance(np.zeros(1), K_free, left) >= 1.0:                                                # the point is a corner of the facet
        u[free] = _boundary_activations(np.zeros(1), K_free, left, W, p, tol, max_iter)
    else:
        u[free], *_ = _solve(np.zeros(1), K_free, left, W, p, tol, max_iter)
    return u


class OnlineSolver:
    """Activations from (state, tau_ref) at run time: Arm gives tau_0 and K, the dual Newton gives u."""

    def __init__(self, arm, p_value: float, rho_max: float = 1.0, tol: float = 1e-9, max_iter: int = 100):
        if arm.n_dofs not in (1, 2):
            raise ValueError("the feasibility check is written for 1 or 2 DoFs")
        if p_value <= 1:
            raise ValueError("p must be > 1: the cost must be strictly convex")
        self.arm = arm
        self.p_value = float(p_value)
        self.rho_max = rho_max
        self.tol = tol
        self.max_iter = max_iter
        self.last = {}   # info of the last solve: rho, saturated, lam, iterations, residual

    def solve(self, tau_0, K, tau_ref, W: float) -> np.ndarray:
        """u (M,) for tau_ref (n,) [Nm]; tau_ref beyond rho_max is moved back to rho_max (rho_max = 1: boundary)."""
        if not 0.0 <= W <= 1.0:
            raise ValueError("W must be in [0, 1]")
        tau_0 = np.ascontiguousarray(tau_0, dtype=np.float64)
        K = np.ascontiguousarray(K, dtype=np.float64)
        tau_ref = np.ascontiguousarray(tau_ref, dtype=np.float64).reshape(len(tau_0))

        rho = _boundary_distance(tau_0, K, tau_ref)
        saturated = rho >= self.rho_max
        tau_target = tau_ref
        if saturated:
            c = tau_0 + 0.5 * K.sum(axis=1)
            tau_target = c + (tau_ref - c) * (self.rho_max / rho)

        if saturated and self.rho_max >= 1.0:       # on the boundary: no Newton (lambda would be infinite)
            u = _boundary_activations(tau_0, K, tau_target, float(W), self.p_value, self.tol, self.max_iter)
            lam, it, res = np.full(len(tau_0), np.nan), 0, float(np.linalg.norm(tau_0 + K @ u - tau_target))
        else:
            u, lam, it, res = _solve(tau_0, K, tau_target, float(W), self.p_value, self.tol, self.max_iter)
        self.last = dict(rho=rho, saturated=saturated, tau_target=tau_target,
                         lam=lam, iterations=it, residual=res)
        return u

    def get_activations(self, W: float, theta, theta_dot, tau) -> np.ndarray:
        """Activations (M,) at the state theta [deg], theta_dot [deg/s] for the torque tau [Nm] (n,)."""
        tau_0, K = self.arm.get_tau0_K(theta, theta_dot)
        return self.solve(tau_0, K, tau, W)
