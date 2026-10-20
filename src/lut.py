import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from src.draw_6x2_heatmaps import draw_heatmaps

N_ACTIVATIONS = 6
ACTIVATION_COLS = [f"activation_value_{i}" for i in range(N_ACTIVATIONS)]
TABLE_COLS = ["W", "angle", "torque_norm", "torque", *ACTIVATION_COLS, "torque_min", "torque_max"]


def from_legacy(ds: xr.Dataset, W: float) -> xr.Dataset:
    """Convert an old LUT (dims angle, torque_norm, activation_idx; no W, no torque) to the current structure."""
    da = ds["activation_value"] if "activation_value" in ds else ds[list(ds.data_vars)[0]]
    ds = ds.rename({"min_torque": "torque_min", "max_torque": "torque_max"}) if "min_torque" in ds.coords else ds
    torque_min = ds["torque_min"].values
    torque_max = ds["torque_max"].values
    torque_norm = ds["torque_norm"].values
    torque = torque_min[:, None] + torque_norm[None, :] * (torque_max - torque_min)[:, None]

    return xr.Dataset(
        data_vars=dict(
            activation_value=(["W", "angle", "torque_norm", "activation_idx"],
                              da.transpose("angle", "torque_norm", "activation_idx").values[None]),
            torque=(["angle", "torque_norm"], torque),
            torque_min=(["angle"], torque_min),
            torque_max=(["angle"], torque_max),
        ),
        coords=dict(
            W=(["W"], [float(W)]),
            angle=(["angle"], ds["angle"].values),
            torque_norm=(["torque_norm"], torque_norm),
            activation_idx=(["activation_idx"], ds["activation_idx"].values),
        ),
    )


def to_table(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten the LUT into rows: W, angle, torque_norm, torque, activation_value_0..5, torque_min, torque_max."""
    df = (
        ds["activation_value"]
        .to_dataframe()
        .reset_index()
        .pivot_table(index=["W", "angle", "torque_norm"], columns="activation_idx", values="activation_value")
        .reset_index()
    )
    df.columns = ["W", "angle", "torque_norm", *ACTIVATION_COLS]

    torque_df = ds[["torque", "torque_min", "torque_max"]].to_dataframe().reset_index()
    df = df.merge(torque_df, on=["angle", "torque_norm"]).dropna()
    return df.sort_values(["W", "angle", "torque_norm"])[TABLE_COLS].reset_index(drop=True)


class LUT:
    def __init__(self, nc_file_path: str, legacy_W: float = None):
        """legacy_W: co-contraction weight of an old-format file (without the W dimension)."""
        self.nc_file_path = nc_file_path
        self.legacy_W = legacy_W
        self.W = None
        self.angle = None
        self.torque_norm = None
        self.torque = None
        self.torque_min = None
        self.torque_max = None
        self.activations_target = None
        self.angle_min = None
        self.angle_max = None
        self.table_df = None
        self.pivot_df = None
        self.lut_data = None

    def load_dataset(self) -> xr.Dataset:
        ds = xr.load_dataset(self.nc_file_path)
        if "W" not in ds.dims:
            if self.legacy_W is None:
                raise ValueError(f"{self.nc_file_path} has the old format: pass legacy_W to LUT()")
            ds = from_legacy(ds, self.legacy_W)
        return ds

    def save(self, ds: xr.Dataset, nc_file_path: str = None) -> None:
        nc_file_path = nc_file_path or self.nc_file_path
        # create temporary file, otherwise on Windows it does not work if the file is open
        temp_file = nc_file_path + ".tmp"
        ds.to_netcdf(temp_file)
        os.replace(temp_file, nc_file_path)

        print(f"File NC saved in: {nc_file_path}")

    def convert_legacy(self, nc_file_path: str = None) -> None:
        """Rewrite an old-format file with the current structure (in place by default)."""
        self.save(self.load_dataset(), nc_file_path)

    def rename_variable(self, old_var_name: str, new_var_name: str) -> None:
        ds = xr.load_dataset(self.nc_file_path)
        if old_var_name in ds:
            ds = ds.rename({old_var_name: new_var_name})
        self.save(ds)

    def to_dataframe(self) -> pd.DataFrame:
        return to_table(self.load_dataset())

    def to_csv(self, csv_file_path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(csv_file_path, index=False)

        print(f"File CSV saved in: {csv_file_path}")
        print(f"Number of valid rows of CSV: {len(df)}")

    def load_data(self, W: float = None) -> None:
        """Load the LUT rows; if W is given, keep only that co-contraction weight (nearest)."""
        df = self.to_dataframe()
        if W is not None:
            W_values = df["W"].unique()
            df = df[df["W"] == W_values[np.abs(W_values - W).argmin()]].reset_index(drop=True)
        self.table_df = df

        self.lut_data = {
            "W": df["W"].values.astype(np.float32),
            "angle": df["angle"].values.astype(np.float32),
            "torque_norm": df["torque_norm"].values.astype(np.float32),
            "torque": df["torque"].values.astype(np.float32),
            "torque_min": df["torque_min"].values.astype(np.float32),
            "torque_max": df["torque_max"].values.astype(np.float32),
            "u_target": df[ACTIVATION_COLS].values.astype(np.float32),
        }

        # set attributes of the class
        self.W = self.lut_data["W"]
        self.angle = self.lut_data["angle"]
        self.torque_norm = self.lut_data["torque_norm"]
        self.torque = self.lut_data["torque"]
        self.torque_min = self.lut_data["torque_min"]
        self.torque_max = self.lut_data["torque_max"]
        self.activations_target = self.lut_data["u_target"]

        self.angle_min = float(self.angle.min())
        self.angle_max = float(self.angle.max())

        # table used only for the heatmaps (first W loaded)
        plot_df = df[df["W"] == df["W"].iloc[0]]
        self.pivot_df = plot_df[["angle", "torque_norm", *ACTIVATION_COLS]].rename(
            columns={col: i for i, col in enumerate(ACTIVATION_COLS)}
        ).reset_index(drop=True)

        print("Dataset LUT uploaded!")
        print(f"W values: {np.unique(self.W)}")
        print(f"LUT matrix dimension: {self.activations_target.shape}")

    def plot_heatmaps(self, angle_req: float = None, tau_req: float = None) -> None:
        # heatmaps of the first W loaded; tau_req is the normalized torque in [0, 1]
        draw_heatmaps(
            df=self.pivot_df,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_req=angle_req,
            tau_req=tau_req,
            cmap='viridis',
            title_prefix='u',
            marker_color='red',
        )
