import itertools
import numpy as np

from scipy.interpolate import interp1d
from src.lut.cost_function import cost_terms, u_rest
from src.utils.plotting import plot_cartesian

AXES = ("angle", "velocity", "torque_norm")


# ----- W: rest activation u*(W) -----
def w_nodes_minimax(n_segments: int, p: float, h_max: float = 1.0, n_fine: int = 20000):
    """W nodes in [0, 1] that minimize the largest linear-interpolation error of u*(W) = u_rest(W, p),
    with n_segments segments no longer than h_max. Returns (nodes, W of max error per segment, max errors)."""
    if n_segments * h_max < 1.0:
        raise ValueError(f"{n_segments} segments of at most {h_max:g} cannot cover [0, 1]")

    # fine grid: geometric near W = 0 (u* has infinite slope there), uniform elsewhere
    w_fine = np.unique(np.concatenate(([0.0], np.geomspace(1e-9, 1.0, n_fine),
                                       np.linspace(0.0, 1.0, n_fine // 4))))
    u_fine = np.array([u_rest(w, p) for w in w_fine])

    def segment_error(a, b):
        """Max of u* - chord on [a, b], and the W where it is."""
        ua, ub = u_rest(a, p), u_rest(b, p)
        inside = (w_fine > a) & (w_fine < b)
        if not inside.any():
            return 0.0, 0.5 * (a + b)
        e = u_fine[inside] - (ua + (ub - ua) / (b - a) * (w_fine[inside] - a))
        i = int(np.argmax(e))
        return float(e[i]), float(w_fine[inside][i])

    def greedy(eps):
        """Each node as far as possible with segment error <= eps and length <= h_max."""
        nodes = [0.0]
        while nodes[-1] < 1.0:
            a = nodes[-1]
            hi = min(1.0, a + h_max)
            if segment_error(a, hi)[0] <= eps:
                nodes.append(hi)
                continue
            lo = a
            for _ in range(50):                      # bisection on the end of the segment
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if segment_error(a, mid)[0] <= eps else (lo, mid)
            nodes.append(lo)
        return np.array(nodes)

    # bisection on eps: the smallest tolerance that needs at most n_segments segments
    eps_lo, eps_hi = 0.0, segment_error(0.0, 1.0)[0]
    for _ in range(40):
        eps = 0.5 * (eps_lo + eps_hi)
        eps_lo, eps_hi = (eps_lo, eps) if len(greedy(eps)) - 1 <= n_segments else (eps, eps_hi)
    nodes = greedy(eps_hi)

    errors, w_max = zip(*(segment_error(a, b) for a, b in zip(nodes[:-1], nodes[1:])))
    return nodes, np.array(w_max), np.array(errors)


def u_rest_error(w_nodes, p: float, n_fine: int = 5000):
    """u*(W) on a fine grid and its linear interpolation between w_nodes: (w_fine, u_exact, u_interpolated)."""
    w_nodes = np.asarray(w_nodes, dtype=float)
    w_fine = np.unique(np.concatenate(([0.0], np.geomspace(1e-6, 1.0, n_fine // 2),
                                       np.linspace(0.0, 1.0, n_fine // 2))))
    u = np.array([u_rest(w, p) for w in w_fine])
    u_lin = np.interp(w_fine, w_nodes, [u_rest(w, p) for w in w_nodes])
    return w_fine, u, u_lin


# ----- angle, velocity, torque: exact activations along one axis -----
def solve_slices(gen, axis: str, x_fine, w_values, angles, velocities, tau_norms):
    """Exact activations along one axis on the fine grid x_fine, for every slice: every combination of
    w_values and of the values of the two other axes (the list of the axis itself is ignored).
    gen: a LutGenerator (only its single-point methods are used, no LUT).
    Returns U (n_slices, n_fine, n_muscles) and one label per slice."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got '{axis}'")
    grids = {"angle": angles, "velocity": velocities, "torque_norm": tau_norms}
    grids[axis] = np.asarray(x_fine, dtype=float)
    others = [name for name in AXES if name != axis]

    # tau_0 depends only on the angle, K on (angle, velocity): computed once with MuJoCo
    tau0_cache, k_cache = {}, {}

    def tau_0(theta):
        if theta not in tau0_cache:
            tau0_cache[theta] = gen.get_tau0(theta)
        return tau0_cache[theta]

    def k(theta, theta_dot):
        if (theta, theta_dot) not in k_cache:
            k_cache[(theta, theta_dot)] = gen.get_K(theta, theta_dot)
        return k_cache[(theta, theta_dot)]

    slices, labels = [], []
    for W in w_values:
        for combo in itertools.product(*(grids[name] for name in others)):
            point = dict(zip(others, (float(c) for c in combo)))
            rows = []
            for x in grids[axis]:
                point[axis] = float(x)
                t0, K = tau_0(point["angle"]), k(point["angle"], point["velocity"])
                tau = gen.get_tau_from_norm(point["torque_norm"], gen.get_tau_min(t0, K), gen.get_tau_max(t0, K))
                rows.append(gen.solve_system(t0, K, tau, float(W))[0])
            slices.append(np.array(rows))
            labels.append(f"W = {W:g}, " + ", ".join(f"{name} = {point[name]:g}" for name in others))
    return np.stack(slices), labels


def grid_error(x_fine, U, nodes) -> dict:
    """Error of a grid along one axis: the values at the nodes (from the fine grid) joined by straight lines,
    against the exact values U (n_slices, n_fine, n_muscles). Returns max, RMSE and the max error at every
    point of x_fine (over slices and muscles)."""
    x_fine = np.asarray(x_fine, dtype=float)
    nodes = np.clip(np.asarray(nodes, dtype=float), x_fine[0], x_fine[-1])
    u_nodes = interp1d(x_fine, U, axis=1)(nodes)                      # (n_slices, n_nodes, n_muscles)
    u_lin = interp1d(nodes, u_nodes, axis=1)(x_fine)                  # straight lines between the nodes
    err = np.abs(u_lin - U)
    return {"max": float(err.max()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_along_axis": err.max(axis=(0, 2))}


def nodes_minimax(x_fine, U, eps: float) -> np.ndarray:
    """Free nodes (on the fine grid) with max error <= eps on every segment, each as far as possible:
    shows where the axis needs more points. U (n_slices, n_fine, n_muscles)."""
    x_fine = np.asarray(x_fine, dtype=float)

    def segment_error(i, j):
        t = (x_fine[i:j + 1] - x_fine[i]) / (x_fine[j] - x_fine[i])
        chord = U[:, i:i + 1, :] * (1 - t)[None, :, None] + U[:, j:j + 1, :] * t[None, :, None]
        return float(np.max(np.abs(U[:, i:j + 1, :] - chord)))

    nodes = [0]
    while nodes[-1] < len(x_fine) - 1:
        i, j = nodes[-1], nodes[-1] + 1
        while j + 1 < len(x_fine) and segment_error(i, j + 1) <= eps:
            j += 1
        nodes.append(j)
    return x_fine[nodes]


def uniform_points_needed(x_fine, U, eps: float, metric: str = "max", odd: bool = False, n_max: int = 501):
    """Smallest number of uniform points on [x_fine[0], x_fine[-1]] with error <= eps (metric "max" or "rmse").
    odd=True: only odd numbers (the middle of the range is a node, e.g. velocity 0). Returns (n, error dict)
    or (None, error dict of n_max) if n_max points are not enough."""
    x_fine = np.asarray(x_fine, dtype=float)
    for n in range(3 if odd else 2, n_max + 1, 2 if odd else 1):
        error = grid_error(x_fine, U, np.linspace(x_fine[0], x_fine[-1], n))
        if error[metric] <= eps:
            return n, error
    return None, error


def tau_norm_nodes(n: int, edge_density: float) -> np.ndarray:
    """torque_norm nodes of LutGenerator.populate_tau_norm: uniform s moved towards 0 and 1."""
    s = np.linspace(0.0, 1.0, n)
    return s - edge_density * np.sin(2 * np.pi * s) / (2 * np.pi)


def torque_points_needed(x_fine, U, eps: float, edge_densities=(0.0, 0.25, 0.5, 0.75),
                         metric: str = "max", n_max: int = 501) -> dict:
    """For every edge_density, the smallest number of torque_norm points with error <= eps.
    x_fine must span [0, 1]. Returns {edge_density: (n or None, error dict)}."""
    result = {}
    for e in edge_densities:
        for n in range(3, n_max + 1):
            error = grid_error(x_fine, U, tau_norm_nodes(n, e))
            if error[metric] <= eps:
                result[e] = (n, error)
                break
        else:
            result[e] = (None, error)
    return result


# ----- plots -----
def plot_cost_terms(p: float, W: float, n_points: int = 500, save_path: str = None, show: bool = True) -> None:
    """J_i(u) and its terms at one (p, W), with the rest activation u*: no LUT needed."""
    u = np.linspace(0.0, 1.0, n_points)
    terms = cost_terms(u, W, p)
    plot_cartesian(title=rf"$J_i(u)$, p = {p:g}, W = {W:g}, $u^*$ = {u_rest(W, p):.3f}",
                   x=u,
                   y=list(terms.values()) + [sum(terms.values())],
                   labels=list(terms.keys()) + [r"$J_i$"],
                   colors=[f"C{i}" for i in range(len(terms))] + ["k"],
                   x_label="u",
                   y_label=r"$J_i(u)$",
                   save_path=save_path,
                   show=show)


def plot_u_rest(w_nodes, p: float, error: bool = False, save_path: str = None, show: bool = True) -> None:
    """u*(W) exact against the straight lines between w_nodes; with error=True, their difference."""
    w_fine, u, u_lin = u_rest_error(w_nodes, p)
    if error:
        plot_cartesian(title=f"Interpolation error of u*(W), p = {p:g}, {len(w_nodes)} W",
                       x=w_fine, y=[u - u_lin],
                       x_label="W", y_label="u* exact - u* interpolated",
                       save_path=save_path, show=show)
    else:
        plot_cartesian(title=f"u*(W), p = {p:g}: exact vs linear interpolation",
                       x=w_fine, y=[u, u_lin],
                       labels=["exact", f"linear interpolation ({len(w_nodes)} W)"],
                       x_label="W", y_label="u*",
                       save_path=save_path, show=show)


def plot_axis_error(x_fine, U, nodes, x_label: str, title: str = None,
                    save_path: str = None, show: bool = True) -> None:
    """Max interpolation error (over slices and muscles) at every point of the axis, for the grid nodes:
    where the error is, so where the grid needs more points."""
    error = grid_error(x_fine, U, nodes)
    plot_cartesian(title=title or f"Interpolation error along {x_label}, {len(nodes)} points: "
                                  f"max {error['max']:.4f}, RMSE {error['rmse']:.5f}",
                   x=np.asarray(x_fine, dtype=float), y=[error["max_along_axis"]],
                   x_label=x_label, y_label="max |u - u interpolated|",
                   save_path=save_path, show=show)
