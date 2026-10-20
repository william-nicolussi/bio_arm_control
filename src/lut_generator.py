import os
import sys
import numpy as np
import xarray as xr

# make the fr_fr_byophysical_arm modules importable (they use flat imports)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARM_DIR = os.path.join(PROJECT_ROOT, "fr_fr_byophysical_arm")
if ARM_DIR not in sys.path:
    sys.path.insert(0, ARM_DIR)

import muscle_co_contraction as mcc


class LUT_generator:
    def __init__(self, W_values, torque_steps: int = mcc.TORQUE_STEPS):
        self.W_values = np.atleast_1d(np.asarray(W_values, dtype=float))
        self.torque_steps = torque_steps
        self.comb_lut = None

    def generate(self) -> xr.Dataset:
        # dims (W, angle, torque_norm, activation_idx); vars activation_value, torque, torque_min, torque_max
        self.comb_lut = mcc.build_comb_lut(self.W_values, self.torque_steps)
        return self.comb_lut

    def save(self, nc_file_path: str) -> None:
        if self.comb_lut is None:
            raise RuntimeError("LUT not generated yet: call generate() first")
        os.makedirs(os.path.dirname(os.path.abspath(nc_file_path)), exist_ok=True)
        # create temporary file, otherwise on Windows it does not work if the file is open
        temp_file = nc_file_path + ".tmp"
        self.comb_lut.to_netcdf(temp_file)
        os.replace(temp_file, nc_file_path)

        print(f"File NC saved in: {nc_file_path}")
