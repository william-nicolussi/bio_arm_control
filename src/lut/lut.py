import os
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import RegularGridInterpolator
from src.utils.metrics import compute_metrics, evaluate_grid
from src.utils.plotting import plot_cartesian, plot_heatmaps, plot_parity
from src.lut.cost_function import cost_terms, cost_i, cost_total, u_rest

# Lut attribute -> name inside the .nc file
NAMES = {"w": "W",
         "theta": "angle",
         "theta_dot": "velocity",
         "tau_norm": "torque_norm",
         "act": "act",
         "tau": "torque",
         "tau_min": "torque_min",
         "tau_max": "torque_max",
         "tau_0": "tau_0",
         "k": "K",
         "lam": "lam",
         "activations": "activations",
         "lut_name": "lut_name",
         "model_name": "model_name",
         "p": "p",
         "muscles_names": "muscles_names",
         "edge_density": "edge_density"}

# Grid dimensions of the LUT, in this order (Lut attribute names)
GRID_DIMS = ("w", "theta", "theta_dot", "tau_norm")

# Variables of the LUT: Lut attribute -> its dimensions (Lut attribute names)
VARIABLES = {"tau":         ("theta", "theta_dot", "tau_norm"),
             "tau_min":     ("theta", "theta_dot"),
             "tau_max":     ("theta", "theta_dot"),
             "tau_0":       ("theta",),
             "k":           ("theta", "theta_dot", "act"),
             "lam":         ("w", "theta", "theta_dot", "tau_norm"),
             "activations": ("w", "theta", "theta_dot", "tau_norm", "act")}

# Metadata, saved as attributes of the file with the same name
ATTRIBUTES = ("lut_name", "model_name", "p", "edge_density")

class Lut:
    def __init__(self, lut_name: str, model_name: str, p: float, muscles_names: list, edge_density: float,
                w: np.ndarray, theta: np.ndarray, theta_dot: np.ndarray, tau: np.ndarray,
                tau_0: np.ndarray, k: np.ndarray, lam: np.ndarray,
                tau_norm: np.ndarray, tau_min: np.ndarray, tau_max: np.ndarray, activations: np.ndarray):

        # metadata
        self.lut_name = lut_name
        self.model_name = model_name # xml file
        self.p = p
        self.muscles_names = muscles_names
        self.edge_density = edge_density

        # Inputs
        self.w = w
        self.theta = theta
        self.theta_dot = theta_dot
        self.tau = tau

        # Other vars. stored
        self.tau_0 = tau_0
        self.k = k
        self.lam = lam

        # Normalized or for normalization
        self.tau_norm = tau_norm
        self.tau_min = tau_min
        self.tau_max = tau_max

        # target
        self.activations = activations

    # ----- load data -----
    @classmethod
    def load(cls, path_file_nc: str) -> "Lut":
        ds = xr.load_dataset(path_file_nc)
        my_lut = Lut(lut_name = ds.attrs[NAMES["lut_name"]],
                     model_name = ds.attrs[NAMES["model_name"]],
                     p = float(ds.attrs[NAMES["p"]]),
                     muscles_names = ds.attrs[NAMES["muscles_names"]].split(","),
                     edge_density = ds.attrs[NAMES["edge_density"]],
                     w = ds[NAMES["w"]].values,
                     theta = ds[NAMES["theta"]].values,
                     theta_dot = ds[NAMES["theta_dot"]].values,
                     tau = ds[NAMES["tau"]].values,
                     tau_0 = ds[NAMES["tau_0"]].values,
                     k = ds[NAMES["k"]].values,
                     lam = ds[NAMES["lam"]].values,
                     tau_norm = ds[NAMES["tau_norm"]].values,
                     tau_min = ds[NAMES["tau_min"]].values,
                     tau_max = ds[NAMES["tau_max"]].values,
                     activations = ds[NAMES["activations"]].values)
        return my_lut

    # ----- evaluation -----
    def evaluate(self, lut) -> dict:
        """Metrics of this LUT, interpolated, at every grid point of another LUT (e.g. the validation LUT):
        same keys as NeuralNetwork.evaluate; n_excluded = points that cannot be interpolated."""
        if lut.lut_name == self.lut_name:
            raise ValueError(f"'{lut.lut_name}' is this LUT: the metrics would be perfect by construction")
        return evaluate_grid(self._interp_grid(lut), lut)

    # ----- save methods -----
    def save_nc(self, dir_nc: str) -> None:
        path_nc = os.path.join(dir_nc, self.lut_name + ".nc")
        # coordinates: the axes of the grid
        coords = {}
        for attr_name in GRID_DIMS:
            file_name = NAMES[attr_name]
            coords[file_name] = np.asarray(getattr(self, attr_name))

        n_act = self.activations.shape[-1]
        coords[NAMES["act"]] = np.arange(1, n_act + 1)

        # variables: each one with the file names of its dimensions
        data_vars = {}
        for attr_name, dims in VARIABLES.items():
            dims_in_file = []
            for dim in dims:
                dims_in_file.append(NAMES[dim])
            file_name = NAMES[attr_name]
            data_vars[file_name] = (tuple(dims_in_file), getattr(self, attr_name))

        # metadata
        attrs = {}
        for attr_name in ATTRIBUTES:
            file_name = NAMES[attr_name]
            attrs[file_name] = getattr(self, attr_name)
        attrs[NAMES["muscles_names"]] = ",".join(self.muscles_names)

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # temporary file + replace: a file on disk is always complete
        os.makedirs(os.path.dirname(os.path.abspath(path_nc)), exist_ok=True)
        temp_file = path_nc + ".tmp"
        ds.to_netcdf(temp_file)
        os.replace(temp_file, path_nc)
        print(f"File NC saved in: {path_nc}")

    def save_csv(self, dir_csv: str) -> None:
        path_csv = os.path.join(dir_csv, self.lut_name + ".csv")

        # grid axes: one value for every point of the grid, shape (n_w, A, V, T)
        w_grid, theta_grid, theta_dot_grid, tau_norm_grid = np.meshgrid(
            self.w, self.theta, self.theta_dot, self.tau_norm, indexing="ij")
        
        grid_shape = w_grid.shape # (n_w, A, V, T)
        n_act = self.activations.shape[-1]

        # every variable to the shape of the grid: ":" where it depends, np.newaxis where it does not
        #                                   w           theta  theta_dot  tau_norm
        tau_grid     = np.broadcast_to(self.tau[np.newaxis, :, :, :],                   grid_shape)
        tau_min_grid = np.broadcast_to(self.tau_min[np.newaxis, :, :, np.newaxis],      grid_shape)
        tau_max_grid = np.broadcast_to(self.tau_max[np.newaxis, :, :, np.newaxis],      grid_shape)
        tau_0_grid   = np.broadcast_to(self.tau_0[np.newaxis, :, np.newaxis, np.newaxis], grid_shape)
        lam_grid     = self.lam      # already (n_w, A, V, T)

        # variables with one value per muscle: (n_w, A, V, T, n_act) -> (n_rows, n_act)
        act_rows = self.activations.reshape(-1, n_act)
        k_grid = np.broadcast_to(self.k[np.newaxis, :, :, np.newaxis, :], grid_shape + (n_act,))
        k_rows = k_grid.reshape(-1, n_act)

        # columns of the table, with the same names as in the .nc file
        columns = {}
        columns[NAMES["w"]] = w_grid.ravel()
        columns[NAMES["theta"]] = theta_grid.ravel()
        columns[NAMES["theta_dot"]] = theta_dot_grid.ravel()
        columns[NAMES["tau"]] = tau_grid.ravel()
        for i in range(n_act):
            column_name = NAMES["act"] + "_" + str(i + 1)
            columns[column_name] = act_rows[:, i]
        columns[NAMES["tau_norm"]] = tau_norm_grid.ravel()
        columns[NAMES["tau_min"]] = tau_min_grid.ravel()
        columns[NAMES["tau_max"]] = tau_max_grid.ravel()
        columns[NAMES["tau_0"]] = tau_0_grid.ravel()
        for i in range(n_act):
            column_name = NAMES["k"] + "_" + str(i + 1)
            columns[column_name] = k_rows[:, i]
        columns[NAMES["lam"]] = lam_grid.ravel()

        df = pd.DataFrame(columns)

        os.makedirs(dir_csv, exist_ok=True)
        df.to_csv(path_csv, index=False)
        print(f"File CSV saved in: {path_csv} ({len(df)} rows)")

    # ----- get values through interpolation -----
    def get_activations(self, w, theta, theta_dot, tau) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, dtype=float))
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        theta_dot = np.atleast_1d(np.asarray(theta_dot, dtype=float))
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        w, theta, theta_dot, tau = np.broadcast_arrays(w, theta, theta_dot, tau)

        # torque limits at (theta, theta_dot), then torque -> torque_norm
        state_axes = (self.theta, self.theta_dot)
        tau_min_interp = RegularGridInterpolator(state_axes, self.tau_min,
                                                 bounds_error=False, fill_value=np.nan)
        tau_max_interp = RegularGridInterpolator(state_axes, self.tau_max,
                                                 bounds_error=False, fill_value=np.nan)
        state_points = np.column_stack([theta, theta_dot])
        tau_min = tau_min_interp(state_points)
        tau_max = tau_max_interp(state_points)
        tau_norm = (tau - tau_min) / (tau_max - tau_min)

        # activations on the regular grid (w, theta, theta_dot, tau_norm)
        if len(self.w) == 1:
            # a single W: no interpolation along w is possible, the W axis is dropped
            if not np.allclose(w, self.w[0]):
                raise ValueError(f"The LUT has only W = {self.w[0]}: requested W = {np.unique(w)}")
            grid_axes = (self.theta, self.theta_dot, self.tau_norm)
            grid_values = self.activations[0]             # (A, V, T, n_act)
            points = np.column_stack([theta, theta_dot, tau_norm])
        else:
            grid_axes = (self.w, self.theta, self.theta_dot, self.tau_norm)
            grid_values = self.activations                # (n_w, A, V, T, n_act)
            points = np.column_stack([w, theta, theta_dot, tau_norm])

        act_interp = RegularGridInterpolator(grid_axes, grid_values,
                                             bounds_error=False, fill_value=np.nan)
        return act_interp(points) # (N, n_act)

    # ----- print -----
    def print_info(self) -> None:
        print(f"Lut name: {self.lut_name}")
        print(f"Model: {self.model_name}")
        print(f"p = {self.p:g}")
        print(f"w = {self.w}")
        print(f"Num angles: {len(self.theta)} from {self.theta[0]:g} to {self.theta[-1]:g} deg")
        print(f"Num velocities: {len(self.theta_dot)} from {self.theta_dot[0]:g} to {self.theta_dot[-1]:g} deg/s")
        print(f"Num torques: {len(self.tau_norm)} per (angle, velocity)")
        print(f"Grid points: {self.lam.size}")   # n_W * A * V * T

        # shape and memory of every array
        total = 0
        for name in ("tau", "tau_min", "tau_max", "tau_0", "k", "lam", "activations"):
            arr = getattr(self, name)
            total += arr.nbytes
            print(f"  {name:<12}: shape {str(arr.shape):<22} {arr.nbytes / 1e6:8.2f} MB")
        print(f"Total in memory: {total / 1e6:.1f} MB")

    # ----- plots -----
    def plot_tau_0(self, save_path: str = None, show: bool = True) -> None:
        plot_cartesian(title=rf"{self.lut_name}: $\tau_0(\theta)$",
                       x=self.theta,
                       y=self.tau_0,
                       x_label=r"$\theta$ [deg]",
                       y_label=r"$\tau_0$ [Nm]",
                       save_path=save_path,
                       show=show)

    def plot_tau_min_max(self, velocity: float = 0.0, save_path: str = None, show: bool = True) -> None:
        v = np.argmin(np.abs(self.theta_dot - velocity))   # nearest velocity of the grid
        plot_cartesian(title=rf"{self.lut_name}: $\tau_{{min}}$, $\tau_{{max}}$ "
                             rf"at $\dot{{\theta}}$ = {self.theta_dot[v]:g} deg/s",
                       x=self.theta,
                       y=[self.tau_min[:, v], self.tau_max[:, v]],
                       labels=[r"$\tau_{min}$", r"$\tau_{max}$"],
                       x_label=r"$\theta$ [deg]",
                       y_label=r"$\tau$ [Nm]",
                       save_path=save_path,
                       show=show)

    def plot_tau_with_tau_min_max(self, velocity: float = 0.0, save_path: str = None,
                                  show: bool = True) -> None:
        v = np.argmin(np.abs(self.theta_dot - velocity))   # nearest velocity of the grid
        n_t = len(self.tau_norm)

        # grid curves: t = 0 and t = n_t - 1 coincide with tau_min and tau_max, skipped
        curves = [self.tau[:, v, t] for t in range(1, n_t - 1)]
        labels = [None] * len(curves)
        colors = ["0.75"] * len(curves) # light gray

        # limits, drawn last so they stay on top
        curves += [self.tau_min[:, v], self.tau_max[:, v]]
        labels += [r"$\tau_{min}$", r"$\tau_{max}$"]
        colors += ["C0", "C3"]

        plot_cartesian(title=rf"{self.lut_name}: $\tau$ grid at $\dot{{\theta}}$ = "
                             rf"{self.theta_dot[v]:g} deg/s, edge density = {self.edge_density:g}",
                       x=self.theta,
                       y=curves,
                       labels=labels,
                       colors=colors,
                       x_label=r"$\theta$ [deg]",
                       y_label=r"$\tau$ [Nm]",
                       save_path=save_path,
                       show=show)

    def plot_heatmaps_k(self, theta_req: float = None, theta_dot_req: float = None,
                        save_path: str = None, show: bool = True) -> None:
        n_act = self.k.shape[-1]

        subplot_titles = []
        for index in range(n_act):
            subplot_titles.append(f"act({index + 1}): {self.muscles_names[index]}")

        plot_heatmaps(title=rf"{self.lut_name}: $K_i(\theta, \dot{{\theta}})$",
                      x=self.theta,
                      y=self.theta_dot,
                      z=[self.k[:, :, i] for i in range(n_act)],
                      subplot_titles=subplot_titles,
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\dot{\theta}$ [deg/s]",
                      colorbar_label="K [Nm]",
                      cmap="bwr",
                      centered=True,
                      x_req=theta_req,
                      y_req=theta_dot_req,
                      marker_color="black",
                      save_path=save_path,
                      show=show)

    def plot_heatmaps_activations(self, w: float, velocity: float = 0.0,
                                  theta_req: float = None, tau_norm_req: float = None,
                                  save_path: str = None, show: bool = True) -> None:
        iw = np.argmin(np.abs(self.w - w))
        v = np.argmin(np.abs(self.theta_dot - velocity))
        n_act = self.activations.shape[-1]

        subplot_titles = []
        for index in range(n_act):
            subplot_titles.append(f"act({index + 1}): {self.muscles_names[index]}")
        plot_heatmaps(title=rf"{self.lut_name}: activations at W = {self.w[iw]:g}, "
                      rf"$\dot{{\theta}}$ = {self.theta_dot[v]:g} deg/s",
                      x=self.theta,
                      y=self.tau_norm,
                      z=[self.activations[iw, :, v, :, i] for i in range(n_act)],
                      subplot_titles=subplot_titles,
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\tau_{norm}$",
                      colorbar_label="activation",
                      vmin=0.0,
                      vmax=1.0,
                      x_req=theta_req,
                      y_req=tau_norm_req,
                      marker_color="red",
                      save_path=save_path,
                      show=show)

    # ----- parity against another LUT -----
    def plot_parity(self, lut, w: float = None, velocity: float = None, max_points: int = 20000,
                    seed: int = 0, save_path: str = None, show: bool = True) -> None:
        """u interpolated from this LUT against the exact u of another LUT (e.g. the validation LUT), one panel
        per muscle: on the whole grid of lut, or only at one W and/or one velocity (the grid values nearest to
        w, velocity). Points this LUT cannot interpolate (NaN) are excluded and counted in the title."""
        if lut.lut_name == self.lut_name:
            raise ValueError(f"'{lut.lut_name}' is this LUT: the parity would be perfect by construction")

        u_pred = self._interp_grid(lut)                                 # (n_W, A, V, T, M), NaN outside
        u_target = lut.activations

        # selection: one W and/or one velocity of the grid of lut (None = all of them)
        selection = []
        iw, iv = slice(None), slice(None)
        if w is not None:
            iw = int(np.argmin(np.abs(lut.w - w)))
            selection.append(f"W = {lut.w[iw]:g}")
        if velocity is not None:
            iv = int(np.argmin(np.abs(lut.theta_dot - velocity)))
            selection.append(f"velocity = {lut.theta_dot[iv]:g} deg/s")
        n_outputs = u_pred.shape[-1]
        u_pred = u_pred[iw, :, iv].reshape(-1, n_outputs)               # (N, M)
        u_target = u_target[iw, :, iv].reshape(-1, n_outputs)

        # points that cannot be interpolated: excluded from metrics and plot
        finite = np.all(np.isfinite(u_pred), axis=1)
        n_excluded = int(np.sum(~finite))
        u_pred, u_target = u_pred[finite], u_target[finite]

        metrics = compute_metrics(u_target, u_pred)
        texts = [f"RMSE {rmse:.3f}   R² {r2:.4f}"
                 for rmse, r2 in zip(metrics["rmse_per_muscle"], metrics["r2_per_muscle"])]

        # points drawn: a random subset, the same for the same seed
        n_points = len(u_target)
        rng = np.random.default_rng(seed)
        drawn = rng.choice(n_points, size=min(max_points, n_points), replace=False)

        plot_parity(title=f"{self.lut_name} interpolated vs {lut.lut_name}: parity, "
                          f"{', '.join(selection) if selection else 'whole grid'}"
                          f"{f' ({n_excluded} points not interpolable)' if n_excluded else ''}",
            x=[[u_target[drawn, i] for i in range(n_outputs)]],
            y=[[u_pred[drawn, i] for i in range(n_outputs)]],
            labels=[f"{self.lut_name} (interpolated)"],
            subplot_titles=[f"u({i + 1}): {name}" for i, name in enumerate(self.muscles_names)],
            texts=[texts],
            x_label=f"u {lut.lut_name}",
            y_label="u interpolated",
            save_path=save_path,
            show=show)

    # ----- plots of the cost (formula in cost_function.py) -----
    def plot_cost(self, w: float = None, n_points: int = 500,
                  save_path: str = None, show: bool = True) -> None:
        """J_i(u) and its terms at one W of the LUT (as Fig. 4 of Parsa et al.).
        w = None -> the first W of the LUT; otherwise the nearest W of the grid."""
        W = float(self.w[0] if w is None else self.w[np.argmin(np.abs(self.w - w))])
        u = np.linspace(0.0, 1.0, n_points)
        terms = cost_terms(u, W, self.p)                 # {label: values}, any number of terms

        curves = list(terms.values()) + [sum(terms.values())]
        labels = list(terms.keys()) + [r"$J_i$"]
        colors = [f"C{i}" for i in range(len(terms))] + ["k"]   # the total in black
        plot_cartesian(title=rf"{self.lut_name}: $J_i(u)$, p = {self.p:g}, W = {W:g}, "
                             rf"$u^*$ = {u_rest(W, self.p):.3f}",
                       x=u,
                       y=curves,
                       labels=labels,
                       colors=colors,
                       x_label="u",
                       y_label=r"$J_i(u)$",
                       save_path=save_path,
                       show=show)

    def plot_cost_all_w(self, save_path: str = None, show: bool = True) -> None:
        """J_i(u) for every W of the LUT, with the minimum u* of each curve in the legend."""
        Lut.plot_cost_curves([(self.p, W) for W in self.w],
                             title=rf"{self.lut_name}: $J_i(u)$ for every W of the LUT",
                             save_path=save_path, show=show)

    def plot_u_rest(self, error: bool = False, w_nodes=None, n_points: int = 5000,
                    save_path: str = None, show: bool = True) -> None:
        """Rest activation u*(W) = u_rest(W, p), the minimum of J_i without torque constraint: exact curve
        against the linear interpolation between w_nodes (default: the W of this LUT); with error=True,
        their difference. The cost comes only from cost_function.py."""
        w_nodes = self.w if w_nodes is None else np.asarray(w_nodes, dtype=float)

        # fine grid: geometric near W = 0 (u* has infinite slope there), uniform elsewhere
        w = np.unique(np.concatenate(([0.0], np.geomspace(1e-6, 1.0, n_points // 2),
                                      np.linspace(0.0, 1.0, n_points // 2))))
        u = np.array([u_rest(w_k, self.p) for w_k in w])
        u_lin = np.interp(w, w_nodes, [u_rest(w_k, self.p) for w_k in w_nodes])   # straight lines between nodes

        if error:
            plot_cartesian(title=f"{self.lut_name}: interpolation error of u*(W), p = {self.p:g}",
                           x=w, y=[u - u_lin],
                           x_label="W", y_label="u* exact - u* interpolated",
                           save_path=save_path, show=show)
        else:
            plot_cartesian(title=f"{self.lut_name}: u*(W), p = {self.p:g}, exact vs linear interpolation",
                           x=w, y=[u, u_lin],
                           labels=["exact", f"linear interpolation ({len(w_nodes)} W)"],
                           x_label="W", y_label="u*",
                           save_path=save_path, show=show)
               
    @staticmethod
    def plot_cost_curves(params, n_points: int = 500, title: str = None,
                         save_path: str = None, show: bool = True) -> None:
        """J_i(u) for every (p, W) in params, e.g. [(2, 0.15), (3, 0.15)] to compare p.
        Static: no LUT needed, the curve depends only on (p, W)."""
        u = np.linspace(0.0, 1.0, n_points)
        curves, labels = [], []
        for p, W in params:
            curves.append(cost_i(u, W, p))
            labels.append(rf"p = {p:g}, W = {W:g}: $u^*$ = {u_rest(W, p):.3f}")
        plot_cartesian(title=title or r"$J_i(u)$ for different (p, W)",
                       x=u,
                       y=curves,
                       labels=labels,
                       x_label="u",
                       y_label=r"$J_i(u)$",
                       save_path=save_path,
                       show=show)

    def plot_heatmap_cost(self, w: float, velocity: float = 0.0,
                          theta_req: float = None, tau_norm_req: float = None,
                          save_path: str = None, show: bool = True) -> None:
        """Total cost J_p of the solutions stored in the LUT, on (theta, tau_norm) at one W and velocity."""
        iw = np.argmin(np.abs(self.w - w))
        v = np.argmin(np.abs(self.theta_dot - velocity))
        J = cost_total(self.activations[iw, :, v, :, :], float(self.w[iw]), self.p)   # (A, T)
        plot_heatmaps(title=rf"{self.lut_name}: $J_p$ at W = {self.w[iw]:g}, "
                            rf"$\dot{{\theta}}$ = {self.theta_dot[v]:g} deg/s",
                      x=self.theta,
                      y=self.tau_norm,
                      z=[J],
                      subplot_titles=[r"$J_p = \sum_i J_i(u_i)$"],
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\tau_{norm}$",
                      colorbar_label=r"$J_p$",
                      x_req=theta_req,
                      y_req=tau_norm_req,
                      marker_color="red",
                      save_path=save_path,
                      show=show)

    # ----- Utils -----
    def _interp_grid(self, lut) -> np.ndarray:
        """u of this LUT interpolated at every grid point of another LUT: (n_W, A, V, T, M), NaN where it
        cannot interpolate (torque outside the interpolated [tau_min, tau_max])."""
        shape = lut.activations.shape[:-1]                                    # (n_W, A, V, T)
        w = np.broadcast_to(lut.w[:, None, None, None], shape).ravel()
        theta = np.broadcast_to(lut.theta[None, :, None, None], shape).ravel()
        theta_dot = np.broadcast_to(lut.theta_dot[None, None, :, None], shape).ravel()
        tau = np.broadcast_to(lut.tau[None, :, :, :], shape).ravel()
        return self.get_activations(w, theta, theta_dot, tau).reshape(lut.activations.shape)