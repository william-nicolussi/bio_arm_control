import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from src import nn_plots

class LUT:
    def __init__(self, nc_file_path: str):
        self.nc_file_path = nc_file_path

        # inputs NN
        self.W = None
        self.angle = None
        self.torque = None

        # for normalization
        self.angle_min = None
        self.angle_max = None
        self.torque_min = None
        self.torque_max = None

        # target (N,6)
        self.activations = None

        # useful
        self.torque_norm = None
        self.angle_grid = None

        # grid axes of the LUT, used by get_activations
        self.W_grid = None
        self.torque_norm_grid = None
        self._interpolator = None  # built at the first call of get_activations

    def rename_variable(self, old_var_name: str, new_var_name: str) -> None:
        ds = xr.load_dataset(self.nc_file_path)
        if old_var_name in ds:
            ds = ds.rename({old_var_name: new_var_name})
        # create temporary file, otherwise on Windows it does not work 
        temp_file = self.nc_file_path + ".tmp"
        ds.to_netcdf(temp_file)
        os.replace(temp_file, self.nc_file_path)
        
        print(f"File NC updated in: {self.nc_file_path}")

    def to_csv(self, csv_file_path: str) -> None:
        """One row per (W, angle, torque_norm), one column per activation (activation_value_0...5)."""
        ds = xr.load_dataset(self.nc_file_path)
 
        # activations: activation_idx becomes columns
        act = ds["activation_value"].transpose("W", "angle", "torque_norm", "activation_idx")
        df = act.to_series().unstack("activation_idx")
        df.columns = [f"activation_value_{int(i)}" for i in df.columns]
        df = df.reset_index()
 
        # torque depends on (angle, torque_norm), the limits only on angle
        df = df.merge(ds["torque"].to_dataframe().reset_index(), on=["angle", "torque_norm"])
        df = df.merge(ds[["torque_min", "torque_max"]].to_dataframe().reset_index(), on="angle")
 
        act_cols = [f"activation_value_{i}" for i in range(act.sizes["activation_idx"])]
        df = df[["W", "angle", "torque_norm", "torque"] + act_cols + ["torque_min", "torque_max"]]
        df = df.sort_values(["W", "angle", "torque_norm"]).dropna()
 
        df.to_csv(csv_file_path, index=False)
 
        print(f"File CSV saved in: {csv_file_path}")
        print(f"Number of valid rows of CSV: {len(df)}")

    def load_data(self, only_W: float = None) -> None:
        ds = xr.load_dataset(self.nc_file_path)

        if only_W is not None:
            ds = ds.sel(W=[only_W], method="nearest")  # list: keeps the W dimension

        # explicit order: rows are (W, angle, torque_norm) whatever the order in the file
        act = ds.activation_value.transpose("W", "angle", "torque_norm", "activation_idx")
        self.activations = act.values.reshape(-1, act.sizes["activation_idx"])

        wg, ang, tn = np.meshgrid(ds.W.values, ds.angle.values, ds.torque_norm.values, indexing="ij")
        self.W = wg.ravel()
        self.angle = ang.ravel()
        self.torque_norm = tn.ravel()

        # torque does not depend on W: repeat it for every W
        torque = ds.torque.transpose("angle", "torque_norm").values
        self.torque = np.broadcast_to(torque, wg.shape).ravel()

        # per-angle tables, used by LUTPreprocessing
        self.angle_grid = ds.angle.values
        self.W_grid = ds.W.values
        self.torque_norm_grid = ds.torque_norm.values
        self._interpolator = None  # data changed: rebuild it when needed
        self.torque_min = ds.torque_min.values
        self.torque_max = ds.torque_max.values
        self.angle_min = float(ds.angle.min())
        self.angle_max = float(ds.angle.max())

        print("Dataset LUT uploaded!")
        print(f"W values: {ds.W.values}")
        print(f"LUT matrix dimension: {self.activations.shape}")

    def torque_to_norm(self, angle, torque) -> tuple:
        """Torque in Nm -> torque_norm at that angle, as the pre-processing does.
        Scalars or arrays. Returns (torque_norm, torque_min, torque_max)."""
        if self.angle_grid is None:
            raise RuntimeError("LUT not loaded: call load_data() first")
        order = np.argsort(self.angle_grid)  # np.interp needs ascending angles
        angles = self.angle_grid[order]
        t_min = np.interp(angle, angles, self.torque_min[order])
        t_max = np.interp(angle, angles, self.torque_max[order])
        return (np.asarray(torque) - t_min) / (t_max - t_min), t_min, t_max

    def _build_interpolator(self) -> None:
        """Trilinear interpolator on the regular grid (W, angle, torque_norm)."""
        axes = [self.W_grid, self.angle_grid, self.torque_norm_grid]
        if len(self.W_grid) < 2:
            raise ValueError(f"The LUT has only W={self.W_grid}: at least 2 W values are "
                             f"needed to interpolate along W (load it without only_W)")
        n_out = self.activations.shape[1]
        cube = self.activations.reshape(*(len(a) for a in axes), n_out)  # rows are (W, angle, torque_norm)

        # RegularGridInterpolator wants ascending axes
        order = [np.argsort(a) for a in axes]
        cube = cube[np.ix_(*order)]
        axes = [a[o] for a, o in zip(axes, order)]

        self._interpolator = RegularGridInterpolator(
            axes, cube, method="linear", bounds_error=False, fill_value=np.nan)

    def get_activations(self, W_req, angle_req, torque_req) -> np.ndarray:
        """Activations u(0...5) interpolated (trilinear) at physical inputs, like NN_validator.predict:
        W, angle (same unit as the LUT) and torque in Nm. Scalars or arrays of the same length.
        Returns (N, 6). NaN outside the LUT domain (W or angle out of range, torque outside
        [torque_min, torque_max]) and next to infeasible (NaN) cells of the LUT."""
        if self.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() first")
        W, angle, torque = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                                  for v in (W_req, angle_req, torque_req)))
        torque_norm, _, _ = self.torque_to_norm(angle, torque)
        if self._interpolator is None:
            self._build_interpolator()
        return self._interpolator(np.column_stack([W, angle, torque_norm]))

    def to_dataframe(self, only_W: float = None) -> pd.DataFrame:
        """One row per LUT point: W, angle, torque, torque_norm, u_target_0...5
        (the format of nn_plots)."""
        if self.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() first")
        df = pd.DataFrame({
            "W": self.W,
            "angle": self.angle,
            "torque": self.torque,
            "torque_norm": self.torque_norm,
        })
        for i in range(self.activations.shape[1]):
            df[f"u_target_{i}"] = self.activations[:, i]
        if only_W is not None:
            df = nn_plots.filter_predictions(df, W=only_W)
        return df

    def plot_heatmaps(self, considered_W: float, angle_req: float = None, tau_req: float = None,
                      save_path: str = None, show: bool = True) -> dict:
        """Heatmaps of the LUT activations over (angle, torque_norm) for one W.
        save_path: a folder, or a full .png path as before."""
        df = self.to_dataframe()
        if not np.isclose(df["W"], considered_W).any():
            raise ValueError(f"W={considered_W} not found. Available: {np.unique(self.W)}")

        folder, file_name = save_path, None
        if save_path is not None and save_path.lower().endswith(".png"):
            folder, file_name = os.path.split(save_path)
            folder = folder or "."
        if file_name is None:
            file_name = f"LUT_heatmaps_W_{considered_W:g}"

        return nn_plots.plot_heatmaps(
            {"LUT": df}, value="u_target", W=considered_W,
            angle_req=angle_req, tau_req=tau_req,
            title=f"LUT activations, W = {considered_W:g}",
            save_path=folder, file_name=file_name, show=show)