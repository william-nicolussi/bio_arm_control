import os

import keras
import numpy as np
import pandas as pd

from src.lut import LUT
from src.lut_preprocessing import LUT_preprocessing
from src.nn_model_from_lut import compute_metrics, N_OUTPUTS
from src.nn_plots import plot_predictions


class NN_validator:
    """Compare a saved Keras model with a LUT (also with W not seen in training)."""

    def __init__(self, path_keras: str, lut: LUT):
        if lut.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() first")

        self.name = os.path.splitext(os.path.basename(path_keras))[0]
        self.lut = lut
        self.model = keras.models.load_model(
            path_keras,
            custom_objects={"LUT_preprocessing": LUT_preprocessing},
        )

        self.u_target = lut.activations.astype("float32")
        self.u_pred = self._predict()   # output with the clip: the one used by the controller
        self.metrics = compute_metrics(self.u_target, self.u_pred)

        W_values = np.unique(lut.W)
        self.W_text = ", ".join(f"{w:g}" for w in W_values)
        self.W_tag = "_".join(f"{w:g}" for w in W_values)

    def _predict(self) -> np.ndarray:
        x = {
            "W": self.lut.W.reshape(-1, 1).astype("float32"),
            "angle": self.lut.angle.reshape(-1, 1).astype("float32"),
            "torque": self.lut.torque.reshape(-1, 1).astype("float32"),
        }
        return self.model.predict(x, batch_size=4096, verbose=0)

    def get_table(self) -> pd.DataFrame:
        """One row per LUT point: inputs, u_target, u_pred, error = u_pred - u_target."""
        df = pd.DataFrame({
            "W": self.lut.W,
            "angle": self.lut.angle,
            "torque": self.lut.torque,
            "torque_norm": self.lut.torque_norm,
        })
        for i in range(N_OUTPUTS):
            df[f"u_target_{i}"] = self.u_target[:, i]
        for i in range(N_OUTPUTS):
            df[f"u_pred_{i}"] = self.u_pred[:, i]
        for i in range(N_OUTPUTS):
            df[f"error_{i}"] = self.u_pred[:, i] - self.u_target[:, i]
        return df

    def plot_predictions(self, show: bool = True, save_path: str = None) -> None:
        plot_predictions(self.u_target, self.u_pred,
                         title=f"{self.name} validation (W={self.W_text})",
                         file_prefix=f"{self.name}_validation_W_{self.W_tag}",
                         save_path=save_path, show=show)