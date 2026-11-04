import os
import json

import numpy as np
import pandas as pd

from src import nn_plots
from src.nn_metrics import MUSCLE_NAMES, n_muscles

try:
    from IPython.display import display
except ImportError:  # outside Jupyter
    display = print


def return_all_model_dirs_in_this_folder(folder_path: str) -> list:
    model_dirs = []
    for root, dirs, files in os.walk(folder_path):
        if any(file.endswith("_metadata.json") for file in files):
            model_dirs.append(root.replace("\\", "/"))
    return sorted(model_dirs)


def print_all_path_in_list(models_to_visualize: list) -> None:
    print("[")
    for model in models_to_visualize:
        print(f"\t'{model}',")
    print("]")


class NN_model_container:
    """A trained model read from its export folder (metadata JSON + history and predictions CSV)."""

    def __init__(self, path_dir: str):

        # paths and files names
        self.path_dir = path_dir
        self.path_json = self.find_metadata_json(path_dir)

        # define parameters
        self.model_name = None
        self.hyperparameters = {}
        self.train_metrics = {}
        self.validation_metrics = {}
        self.path_csv_history = None
        self.path_csv_preds = None
        self.history = None
        self.predictions = None

        # set parameters
        self.load_parameters()

    @staticmethod
    def find_metadata_json(path_dir: str) -> str:
        json_files = [f for f in os.listdir(path_dir) if f.endswith("_metadata.json")]
        if len(json_files) != 1:
            raise FileNotFoundError(
                f"Expected 1 '*_metadata.json' in {path_dir}, found {len(json_files)}: {json_files}"
            )
        return os.path.join(path_dir, json_files[0])

    def load_parameters(self):
        with open(self.path_json, 'r') as f:
            json_data = json.load(f)

        self.model_name = json_data.get("model_name")
        self.hyperparameters = json_data.get("hyperparameters") or {}
        self.train_metrics = json_data.get("train_metrics") or {}
        self.validation_metrics = json_data.get("validation_metrics") or {}

        # csv names are written in the json, under "files"
        files = json_data.get("files") or {}
        self.path_csv_history = os.path.join(self.path_dir, files["history"])
        self.path_csv_preds = os.path.join(self.path_dir, files["predictions"])
        self.history = pd.read_csv(self.path_csv_history)
        self.predictions = pd.read_csv(self.path_csv_preds)

    @property
    def label(self) -> str:
        return f"{self.model_name} ({self.hyperparameters.get('num_parameters')} param)"

    def get_predictions(self, W=None, angle_range: tuple = None,
                        torque_norm_range: tuple = None) -> pd.DataFrame:
        return nn_plots.filter_predictions(self.predictions, W, angle_range, torque_norm_range)

    def n_muscles(self) -> int:
        return n_muscles(self.predictions)


class NN_models_visualizer:
    """Compare models. Every element of models_to_visualize can be:
      - the path of an export folder (str)  -> NN_model_container (training LUT, history, metrics)
      - an NN_model_container
      - an NN_validator                     -> predictions on its validation LUT (no history)
    Models compared together should share the same LUT (same rows), otherwise only the
    target of the first one is drawn in the series plots."""
    MUSCLE_NAMES = MUSCLE_NAMES

    def __init__(self, models_to_visualize: list):
        self.list_models = []
        for model in models_to_visualize:
            self.add_model(model)

    def add_model(self, model) -> None:
        """A path of an export folder, an NN_model_container or an NN_validator."""
        if isinstance(model, (str, os.PathLike)):
            model = NN_model_container(os.fspath(model))
        if not hasattr(model, "predictions") or not hasattr(model, "label"):
            raise TypeError(f"Cannot visualize {type(model).__name__}: "
                            f"expected a path, an NN_model_container or an NN_validator")
        self.list_models.append(model)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _name(model) -> str:
        return getattr(model, "model_name", None) or model.name

    @staticmethod
    def _is_validator(model) -> bool:
        return not isinstance(model, NN_model_container)

    def _frames(self) -> dict:
        """{label: predictions}; repeated labels get a suffix, so no model is lost."""
        frames = {}
        for model in self.list_models:
            label, k = model.label, 2
            while label in frames:
                label, k = f"{model.label} #{k}", k + 1
            frames[label] = model.predictions
        return frames

    @staticmethod
    def _W_suffix(W) -> str:
        """Suffix of the file names, as before: _W_0_5 or _all_W."""
        if W is None:
            return "_all_W"
        return "_W_" + "_".join(str(float(w)).replace(".", "_") for w in np.atleast_1d(W))

    # ------------------------------------------------------------------ table

    def print_table(self, metrics_in_table: list = None, split: str = "train") -> pd.DataFrame:
        """Scalar metrics, one row per model. split = "train" or "validation" for the models
        read from folders; an NN_validator always shows its metrics on its validation LUT."""

        def metrics_of(model) -> dict:
            if self._is_validator(model):
                return model.metrics
            return model.validation_metrics if split == "validation" else model.train_metrics

        def data_of(model) -> str:
            if self._is_validator(model):
                return f"validation W={model.W_text}"
            return split

        # if method called empty, add all scalar metrics (keeping json order)
        if not metrics_in_table:
            metrics_in_table = []
            for model in self.list_models:
                for name, value in metrics_of(model).items():
                    if isinstance(value, (int, float)) and name not in metrics_in_table:
                        metrics_in_table.append(name)

        # populate table
        data = []
        for model in self.list_models:
            metrics = metrics_of(model)
            row_data = {"Model Name": self._name(model), "Data": data_of(model)}
            for metric in metrics_in_table:
                row_data[metric] = metrics.get(metric, None)
            data.append(row_data)

        df = pd.DataFrame(data)
        if display is not print:
            sci_cols = [c for c in ("mse",) if c in df.columns]   # very small values
            styler = df.style.format(precision=4, na_rep="N/A")
            if sci_cols:
                styler = styler.format("{:.3e}", subset=sci_cols, na_rep="N/A")
            display(styler)
        else:
            display(df)
        return df

    # ------------------------------------------------------------------ plots

    def plot_history(self, metric: str = "loss", save_path: str = None, log_scale: bool = True,
                     show: bool = True) -> None:
        histories = {model.label: model.history for model in self.list_models
                     if getattr(model, "history", None) is not None}
        skipped = [self._name(m) for m in self.list_models if getattr(m, "history", None) is None]
        if skipped:
            print(f"[warning] no history (NN_validator) for: {', '.join(skipped)}")
        if not histories:
            return
        nn_plots.plot_history(histories, metric=metric, log_scale=log_scale,
                              save_path=save_path, file_name=f"{metric}_comparison", show=show)

    def plot_u_pred_u_target(self, W=None, angle_range: tuple = None,
                             torque_norm_range: tuple = None, muscles: list = None,
                             save_path: str = None, show: bool = True) -> None:
        nn_plots.plot_series(self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
                             save_path=save_path, file_name="u_pred_u_target", show=show)

    def plot_error(self, W=None, angle_range: tuple = None,
                   torque_norm_range: tuple = None, muscles: list = None,
                   save_path: str = None, show: bool = True) -> None:
        """u_pred - u_target along the samples, one subplot per muscle."""
        nn_plots.plot_series(self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
                             error=True, save_path=save_path, file_name="u_pred_error", show=show)

    def plot_u_raw_u_pred(self, W=None, angle_range: tuple = None,
                          torque_norm_range: tuple = None, muscles: list = None,
                          save_path: str = None, show: bool = True) -> None:
        nn_plots.plot_raw_vs_pred(self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
                                  save_path=save_path, file_name="u_raw_u_pred", show=show)

    def plot_parity(self, W=None, angle_range: tuple = None,
                    torque_norm_range: tuple = None, muscles: list = None,
                    max_points: int = 20000, ncols: int = 3, point_size: float = 4,
                    alpha: float = None, save_path: str = None, seed: int = 0,
                    show: bool = True) -> None:
        """Parity plot (u_pred vs u_target), one subplot per muscle, one color per model.
        RMSE and R2 on ALL filtered samples; at most max_points samples per model are drawn."""
        nn_plots.plot_parity(self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
                             max_points=max_points, ncols=ncols, point_size=point_size,
                             alpha=alpha, seed=seed, save_path=save_path,
                             file_name=f"parity{self._W_suffix(W)}", show=show)

    def plot_error_hist(self, W=None, angle_range: tuple = None,
                        torque_norm_range: tuple = None, muscles: list = None,
                        bins: int = 60, log: bool = False,
                        save_path: str = None, show: bool = True) -> None:
        nn_plots.plot_error_hist(self._frames(), W, angle_range, torque_norm_range, muscles=muscles,
                                 bins=bins, log=log, save_path=save_path,
                                 file_name=f"error_hist{self._W_suffix(W)}", show=show)

    def plot_heatmaps(self, W: float, value: str = "error", models: list = None,
                      save_path: str = None, show: bool = True) -> None:
        """Heatmaps of value (error, u_pred, u_target, u_raw) for one W: one figure per model.
        models: indices or names to draw (None = all). For the marker at a requested point use
        the plot_heatmaps method of NN_validator / NN_model_from_LUT (they convert Nm)."""
        for k, model in enumerate(self.list_models):
            name = self._name(model)
            if models is not None and k not in models and name not in models:
                continue
            nn_plots.plot_heatmaps({model.label: model.predictions}, value=value, W=W,
                                   save_path=save_path, show=show,
                                   file_name=f"{name}_{value}_heatmaps_W_{W:g}")