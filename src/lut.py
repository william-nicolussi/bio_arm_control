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
        ds = xr.load_dataset(self.nc_file_path)
        
        df = ds.to_dataframe()
        df = df.reset_index()
        df = df.dropna()
        
        df.to_csv(csv_file_path, index=False)
        
        print(f"File CSV saved in: {csv_file_path}")
        print(f"Number of valid rows of CSV: {len(df)}")

    def load_data(self, only_W: float = None) -> None:
        ds = xr.load_dataset(self.nc_file_path)

        if only_W is not None:
            ds = ds.sel(W=only_W)

        # activations: (N, 6)
        act = ds.activation_value.values
        self.activations = act.reshape(-1, act.shape[-1])

        # input grids, same row order as activations
        if only_W is not None:
            ang, tn = np.meshgrid(
                ds.angle.values, ds.torque_norm.values, indexing='ij'
            )
            self.W = np.full(len(self.activations), only_W)
        else:
            wg, ang, tn = np.meshgrid(
                ds.W.values, ds.angle.values, ds.torque_norm.values,
                indexing='ij'
            )
            self.W = wg.ravel()

        self.angle = ang.ravel()
        self.torque_norm = tn.ravel()

        # physical torque: (angle, torque_norm), tiled for each W
        n_W = 1 if only_W is not None else len(ds.W)
        self.torque = np.tile(ds.torque.values.ravel(), n_W)
        self.torque_min = ds.torque_min.values
        self.torque_max = ds.torque_max.values

        # angle range
        self.angle_min = float(ds.angle.min())
        self.angle_max = float(ds.angle.max())

        print("Dataset LUT uploaded!")
        print(f"LUT matrix dimension: {self.activations.shape}")


    def plot_heatmaps(self, considered_W: float, angle_req: float = None, tau_req: float = None) -> None:
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
            cmap='viridis',
            title_prefix='u',
            marker_color='red',
        )