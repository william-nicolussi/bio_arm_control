import os

import keras
import numpy as np
import pandas as pd

from src import nn_plots
from src.lut import LUT
from src.lut_preprocessing import LUT_preprocessing
from src.nn_metrics import N_OUTPUTS, compute_metrics


def torque_to_norm(lut: LUT, angle: float, torque: float) -> tuple:
    """Torque in Nm -> torque_norm at that angle. Kept for compatibility: see LUT.torque_to_norm.
    Returns (torque_norm, torque_min, torque_max)."""
    tau, t_min, t_max = lut.torque_to_norm(angle, torque)
    return float(tau), float(t_min), float(t_max)


class NN_validator:
    """Compare a saved Keras model with a LUT (also with W not seen in training).

    Builds the predictions DataFrame once (self.predictions) and draws it with nn_plots.
    Several validators on the same LUT can be compared by NN_models_visualizer through
    their `label` and `predictions`."""

    def __init__(self, path_keras: str, lut: LUT, label: str = None):
        if lut.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() first")

        self.name = os.path.splitext(os.path.basename(path_keras))[0]
        self.model = keras.models.load_model(
            path_keras,
            custom_objects={"LUT_preprocessing": LUT_preprocessing},
        )
        self.num_parameters = self.model.count_params()
        self._setup(lut, label or f"{self.name} ({self.num_parameters} param)")

    def _setup(self, lut: LUT, label: str) -> None:
        """Common part of the constructor: predictions on the LUT, metrics and table."""
        self.lut = lut
        self.label = label

        W_values = np.unique(lut.W)
        self.W_text = ", ".join(f"{w:g}" for w in W_values)
        self.W_tag = "_".join(f"{w:g}" for w in W_values)

        self.u_target = lut.activations.astype("float32")
        self.u_pred, self.u_raw = self._predict()  # u_pred has the clip: the one used by the controller
        self.metrics = compute_metrics(self.u_target, self.u_pred)
        self.predictions = self.get_table()

    # ------------------------------------------------------------------ data

    def _inputs(self, W, angle, torque) -> dict:
        return {
            "W": np.asarray(W, dtype="float32").reshape(-1, 1),
            "angle": np.asarray(angle, dtype="float32").reshape(-1, 1),
            "torque": np.asarray(torque, dtype="float32").reshape(-1, 1),
        }

    def _predict(self) -> tuple:
        """u_pred (output of the model) and u_raw (output of the layer 'u_raw', before the clip).
        u_raw is None if the saved model has no layer with that name."""
        x = self._inputs(self.lut.W, self.lut.angle, self.lut.torque)
        u_pred = self.model.predict(x, batch_size=4096, verbose=0)
        try:
            raw_model = keras.Model(self.model.input, self.model.get_layer("u_raw").output)
            u_raw = raw_model.predict(x, batch_size=4096, verbose=0)
        except ValueError:
            u_raw = None
        return u_pred, u_raw

    def predict(self, W, angle, torque) -> np.ndarray:
        """u(0...5) for physical inputs (scalars or arrays of the same length). Returns (N, 6)."""
        W, angle, torque = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype="float32"))
                                                  for v in (W, angle, torque)))
        return self.model.predict(self._inputs(W, angle, torque), verbose=0)

    def get_table(self) -> pd.DataFrame:
        """One row per LUT point: inputs, u_target, u_raw (if available), u_pred,
        error = u_pred - u_target."""
        df = pd.DataFrame({
            "W": self.lut.W,
            "angle": self.lut.angle,
            "torque": self.lut.torque,
            "torque_norm": self.lut.torque_norm,
        })
        for i in range(N_OUTPUTS):
            df[f"u_target_{i}"] = self.u_target[:, i]
        if self.u_raw is not None:
            for i in range(N_OUTPUTS):
                df[f"u_raw_{i}"] = self.u_raw[:, i]
        for i in range(N_OUTPUTS):
            df[f"u_pred_{i}"] = self.u_pred[:, i]
        for i in range(N_OUTPUTS):
            df[f"error_{i}"] = self.u_pred[:, i] - self.u_target[:, i]
        return df

    # ----------------------------------------------------------------- plots

    def _frames(self) -> dict:
        return {self.label: self.predictions}

    def _title(self, base: str, W, angle_range, torque_norm_range) -> str:
        filters = nn_plots.filters_title(W, angle_range, torque_norm_range)
        return f"{self.label}: {base}" + (filters or f"  (validation W={self.W_text})")

    def _prefix(self, W) -> str:
        """Prefix of the saved files: {name}_validation_W_{W values}, as before the refactoring."""
        W_tag = self.W_tag if W is None else "_".join(f"{w:g}" for w in np.atleast_1d(W))
        return f"{self.name}_validation_W_{W_tag}"

    def plot_series(self, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                    muscles: list = None, error: bool = False,
                    show: bool = True, save_path: str = None, file_name: str = None) -> None:
        base = "error = u_pred - u_target" if error else "u_target vs u_pred"
        nn_plots.plot_series(
            self._frames(), W, angle_range, torque_norm_range, muscles=muscles, error=error,
            title=self._title(base, W, angle_range, torque_norm_range),
            save_path=save_path, show=show,
            file_name=file_name or f"{self._prefix(W)}_{'error' if error else 'series'}")

    def plot_parity(self, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                    muscles: list = None, max_points: int = 20000,
                    show: bool = True, save_path: str = None, file_name: str = None) -> None:
        nn_plots.plot_parity(
            self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
            max_points=max_points,
            title=self._title("Parity: u_pred vs u_target", W, angle_range, torque_norm_range),
            save_path=save_path, show=show, file_name=file_name or f"{self._prefix(W)}_parity")

    def plot_error_hist(self, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                        muscles: list = None, show: bool = True, save_path: str = None,
                        file_name: str = None) -> None:
        nn_plots.plot_error_hist(
            self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
            title=self._title("error distribution", W, angle_range, torque_norm_range),
            save_path=save_path, show=show, file_name=file_name or f"{self._prefix(W)}_hist")

    def plot_raw_vs_pred(self, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                         muscles: list = None, show: bool = True, save_path: str = None,
                         file_name: str = None) -> None:
        nn_plots.plot_raw_vs_pred(
            self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
            title=self._title("u_raw vs u_pred", W, angle_range, torque_norm_range),
            save_path=save_path, show=show, file_name=file_name or f"{self._prefix(W)}_u_raw_u_pred")

    def plot_predictions(self, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                         show: bool = True, save_path: str = None) -> None:
        """The standard set: series, error, parity, error histogram (same file names as before)."""
        args = (W, angle_range, torque_norm_range)
        self.plot_series(*args, show=show, save_path=save_path)
        self.plot_series(*args, error=True, show=show, save_path=save_path)
        self.plot_parity(*args, show=show, save_path=save_path)
        self.plot_error_hist(*args, show=show, save_path=save_path)

    def plot_heatmaps(self, W: float = None, value: str = "error",
                      angle_req: float = None, torque_req: float = None,
                      show: bool = True, save_path: str = None, file_name: str = None) -> dict:
        """Heatmaps of value (error, u_pred, u_target, u_raw) over (angle, torque_norm) for one W.
        W can be omitted if the LUT has one W only. torque_req in Nm."""
        if W is None:
            W_values = np.unique(self.lut.W)
            if len(W_values) != 1:
                raise ValueError(f"The LUT has more W: pass W= (available: {W_values})")
            W = float(W_values[0])

        tau_req = None
        if angle_req is not None and torque_req is not None:
            tau_req, t_min, t_max = torque_to_norm(self.lut, angle_req, torque_req)
            if not 0.0 <= tau_req <= 1.0:
                print(f"Torque {torque_req} Nm is outside [{t_min:.2f}, {t_max:.2f}] Nm "
                      f"at {angle_req} deg: no marker")
                tau_req = None
            else:
                # the heatmap shows the nearest grid point; the network gives the exact value
                u_exact = self.predict(W, angle_req, torque_req)[0]
                print(f"{self.label} at exactly (W={W:g}, angle={angle_req}, torque={torque_req} Nm, "
                      f"torque_norm={tau_req:.3f}): " + ", ".join(f"{v:.4f}" for v in u_exact))

        return nn_plots.plot_heatmaps(
            self._frames(), value=value, W=W, angle_req=angle_req, tau_req=tau_req,
            save_path=save_path, show=show,
            file_name=file_name or f"{self.name}_validation_{value}_heatmaps_W_{W:g}")


class LUT_validator(NN_validator):
    """Baseline without network: the training LUT interpolated (trilinear in W, angle, torque_norm)
    on the points of a validation LUT. Same interface as NN_validator (metrics, predictions,
    plots), so it can be compared with the networks in NN_models_visualizer.

    train_lut: the LUT the networks were trained on (several W values)
    lut:       the validation LUT (e.g. a W not seen in training)"""

    def __init__(self, train_lut: LUT, lut: LUT, label: str = None):
        if train_lut.activations is None or lut.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() on both LUTs first")

        W_train = np.unique(train_lut.W)
        seen = [w for w in np.unique(lut.W) if np.isclose(W_train, w).any()]
        if seen:
            print(f"[warning] W={seen} is also in the training LUT: "
                  f"the interpolation there is exact, the comparison is not fair")

        self.train_lut = train_lut
        self.name = "LUT_interp_" + os.path.splitext(os.path.basename(train_lut.nc_file_path))[0]
        self.model = None
        self.num_parameters = int(train_lut.activations.size)  # values stored in the table
        self._setup(lut, label or f"LUT interpolation ({self.num_parameters} values)")

    def _predict(self) -> tuple:
        """Interpolated activations on the validation points (same physical inputs as the NN)."""
        u_pred = self.predict(self.lut.W, self.lut.angle, self.lut.torque)
        return u_pred, None  # no u_raw: the interpolation of values in [0, 1] needs no clip

    def predict(self, W, angle, torque) -> np.ndarray:
        """u(0...5) interpolated from the training LUT. Returns (N, 6), NaN outside its domain."""
        return self.train_lut.get_activations(W, angle, torque).astype("float32")