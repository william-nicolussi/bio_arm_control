import os
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import RegularGridInterpolator

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

    # ----- plots -----
    def plot_tau_0():
        print("to do")

    def plot_tau_min_max():
        print("to do")

    def plot_heatmaps(self, considered_W: float, angle_req: float = None, tau_req: float = None,
                      save_path: str = None, show: bool = True, velocity: float = 0.0) -> dict:
        print("to do")

    def plot_heatmaps_k():
        print("to do")

    