import os
import numpy as np
import xarray as xr
import pandas as pd

from src.draw_6x2_heatmaps import draw_heatmaps

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
        self.torque_min = ds.torque_min.values
        self.torque_max = ds.torque_max.values
        self.angle_min = float(ds.angle.min())
        self.angle_max = float(ds.angle.max())

        print("Dataset LUT uploaded!")
        print(f"W values: {ds.W.values}")
        print(f"LUT matrix dimension: {self.activations.shape}")

    def plot_heatmaps(self, considered_W: float, angle_req: float = None, tau_req: float = None, save_path: str=None, show: bool = True) -> None:
        mask = self.W == considered_W
        if not np.any(mask):
            raise ValueError(
                f"W={considered_W} not found. Available: {np.unique(self.W)}"
            )

        angle = self.angle[mask]
        torque_norm = self.torque_norm[mask]
        activations = self.activations[mask]

        # build the DataFrame that draw_heatmaps expects
        df = pd.DataFrame({
            'angle': angle,
            'torque_norm': torque_norm,
        })
        for i in range(activations.shape[1]):
            df[i] = activations[:, i]

        draw_heatmaps(
            df=df,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_req=angle_req,
            tau_req=tau_req,
            save_path = save_path,
            show=show,
            cmap='viridis',
            title_prefix='u',
            marker_color='red',
        )