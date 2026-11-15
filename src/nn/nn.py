import json
import os

import keras
import numpy as np
import pandas as pd
import xarray as xr

from src.nn.nn_preprocessing import NnPreprocessing  # noqa: F401  registers the layer, needed by load_model
from src.utils.plotting import plot_cartesian, plot_heatmaps, plot_parity

INPUT_NAMES = ("W", "angle", "velocity", "torque")
PREDICT_BATCH = 65536   # rows per call of predict: same result, bounded memory
DIRECT_CALL_MAX = 1024  # up to this many rows the model is called directly, without predict

def grid_inputs(lut) -> dict:
    """Inputs of the network (physical units) at every grid point (W, angle, velocity, torque_norm) of a LUT.
    Rows follow the order of the LUT arrays: an output reshaped to lut.activations.shape is back on the grid."""
    grid_shape = lut.activations.shape[:-1]                       # (n_W, A, V, T)

    def column(values):
        # one axis (or table) of the LUT -> value at every grid point -> column (N, 1)
        return np.broadcast_to(values, grid_shape).reshape(-1, 1).astype("float32")

    return {
        "W": column(lut.w[:, None, None, None]),
        "angle": column(lut.theta[None, :, None, None]),
        "velocity": column(lut.theta_dot[None, None, :, None]),
        "torque": column(lut.tau[None, :, :, :]),
    }

def compute_metrics(u_target: np.ndarray, u_pred: np.ndarray) -> dict:
    """Errors of u_pred against u_target, both (N, M): global values and one value per muscle."""
    error = u_pred - u_target
    deviation = u_target - u_target.mean(axis=0)              # target minus its mean, muscle by muscle

    # global (all muscles together)
    mse = float(np.mean(error ** 2))
    r2 = 1.0 - np.sum(error ** 2) / np.sum(deviation ** 2)

    # per muscle
    rmse_per_muscle = np.sqrt(np.mean(error ** 2, axis=0))
    r2_per_muscle = 1.0 - np.sum(error ** 2, axis=0) / np.sum(deviation ** 2, axis=0)
    # FIT (MATLAB NRMSE fit): 100 % = perfect, 0 % = as good as a constant equal to the mean
    fit_per_muscle = 100.0 * (1.0 - np.linalg.norm(error, axis=0) / np.linalg.norm(deviation, axis=0))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "max_error": float(np.max(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(r2),
        "fit_pct": float(np.mean(fit_per_muscle)),
        "rmse_per_muscle": rmse_per_muscle.tolist(),
        "bias_per_muscle": np.mean(error, axis=0).tolist(),
        "max_error_per_muscle": np.max(np.abs(error), axis=0).tolist(),
        "r2_per_muscle": r2_per_muscle.tolist(),
        "fit_pct_per_muscle": fit_per_muscle.tolist(),
    }

def _to_builtin(value):
    """json.dump cannot write numpy types: numpy scalar -> Python number, array -> list."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")

class NeuralNetwork:
    def __init__(self, nn_name: str, model: keras.Model,
                 lut_name: str, p: float, muscles_names: list,
                 num_hidden_layers: int, hidden_units: int, hidden_function: str,
                 output_function: str, seed: int,
                 epochs: int, batch_size: int, learning_rate: float, lr_final_factor: float,
                 history: dict, train_metrics: dict = None, validation_metrics: dict = None,
                 validation_lut_name: str = None):

        # metadata
        self.nn_name = nn_name
        self.model = model # Keras model with the clip: (W, angle, velocity, torque) -> u

        # origin: the LUT used for training
        self.lut_name = lut_name
        self.validation_lut_name = validation_lut_name
        self.p = p
        self.muscles_names = muscles_names # titles of the plots

        # structure
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.hidden_function = hidden_function
        self.output_function = output_function
        self.seed = seed
        self.num_parameters = self.model.count_params()

        # training
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate # initial lr of the cosine decay
        self.lr_final_factor = lr_final_factor
        self.history = history # {"loss": [...], "mae": [...], "learning_rate": [...]}, one value per epoch

        # metrics
        self.train_metrics = train_metrics if train_metrics is not None else {}
        self.validation_metrics = validation_metrics if validation_metrics is not None else {}
    
    # ----- load -----
    @classmethod
    def load(cls, dir_nn: str) -> "NeuralNetwork":
        json_files = [f for f in os.listdir(dir_nn) if f.endswith("_metadata.json")]
        if len(json_files) != 1:
            raise FileNotFoundError(f"Expected 1 '*_metadata.json' in {dir_nn}, found {len(json_files)}: {json_files}")
        with open(os.path.join(dir_nn, json_files[0]), encoding="utf-8") as f:
            metadata = json.load(f)
        files = metadata["files"]
        structure = metadata["structure"]
        training = metadata["training"]

        # model only for prediction: training is done by NnGenerator, so no compile
        model = keras.models.load_model(os.path.join(dir_nn, files["model"]), compile=False)

        # history: CSV (one row per epoch) -> {"loss": [...], "mae": [...], "learning_rate": [...]}
        history_df = pd.read_csv(os.path.join(dir_nn, files["history"]))
        history = {column: history_df[column].tolist() for column in history_df.columns if column != "epoch"}

        nn = cls(nn_name=metadata["nn_name"],
                 model=model,
                 lut_name=metadata["lut_name"],
                 p=metadata["p"],
                 muscles_names=metadata["muscles_names"],
                 num_hidden_layers=structure["num_hidden_layers"],
                 hidden_units=structure["hidden_units"],
                 hidden_function=structure["hidden_function"],
                 output_function=structure["output_function"],
                 seed=structure["seed"],
                 epochs=training["epochs"],
                 batch_size=training["batch_size"],
                 learning_rate=training["learning_rate"],
                 lr_final_factor=training["lr_final_factor"],
                 history=history,
                 train_metrics=metadata["train_metrics"],
                 validation_metrics=metadata.get("validation_metrics", {}),
                 validation_lut_name=metadata.get("validation_lut_name"))

        # the files must describe the same network
        if nn.num_parameters != structure["num_parameters"]:
            raise ValueError(f"{files['model']} has {nn.num_parameters} parameters, "
                             f"the metadata say {structure['num_parameters']}")
        return nn

    # ----- save -----
    def save(self, dir_nn: str, lut=None) -> None:
        """Save the network in dir_nn: {nn_name}.keras, {nn_name}_history.csv, {nn_name}_metadata.json.
        With lut (the training LUT) also {nn_name}_predictions.nc.
        The metadata are written last: a folder with a metadata file always contains a complete network."""
        os.makedirs(dir_nn, exist_ok=True)
        files = {"model": f"{self.nn_name}.keras",
                 "history": f"{self.nn_name}_history.csv"}
        if lut is not None:
            files["predictions"] = f"{self.nn_name}_predictions.nc"
            self._save_predictions_nc(lut, os.path.join(dir_nn, files["predictions"]))

        # Keras model: pre-processing + network + clip
        self.model.save(os.path.join(dir_nn, files["model"]))

        # history: one row per epoch
        history_df = pd.DataFrame(self.history)
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
        history_df.to_csv(os.path.join(dir_nn, files["history"]), index=False)

        # metadata: everything that is not in the model
        metadata = {
            "nn_name": self.nn_name,
            "lut_name": self.lut_name,
            "p": self.p,
            "muscles_names": self.muscles_names,
            "structure": {
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_units": self.hidden_units,
                "hidden_function": self.hidden_function,
                "output_function": self.output_function,
                "seed": self.seed,
                "num_parameters": self.num_parameters,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "lr_final_factor": self.lr_final_factor,
            },
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "validation_lut_name": self.validation_lut_name,
            "keras_version": keras.__version__,
            "keras_backend": keras.backend.backend(),
            "files": files,
        }
        path_json = os.path.join(dir_nn, f"{self.nn_name}_metadata.json")
        temp_file = path_json + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, default=_to_builtin)
        os.replace(temp_file, path_json)
        print(f"Neural network saved in: {dir_nn}")

    # ----- evaluation -----
    def evaluate(self, lut) -> dict:
        """Metrics of the network on every grid point of a LUT (the training LUT or another one):
        errors on u (global and per muscle) and torque error |tau_0 + K·u - tau_ref| [Nm]."""
        u_pred = np.clip(self._predict_grid(lut), 0.0, 1.0)                # (n_W, A, V, T, M)
        n_outputs = u_pred.shape[-1]
        metrics = compute_metrics(lut.activations.reshape(-1, n_outputs), u_pred.reshape(-1, n_outputs))

        torque_error = np.abs(self._torque_grid(lut, u_pred) - lut.tau[None, :, :, :])
        metrics["torque_err_mean_Nm"] = float(np.mean(torque_error))
        metrics["torque_err_p95_Nm"] = float(np.percentile(torque_error, 95))
        metrics["torque_err_max_Nm"] = float(np.max(torque_error))
        return metrics

    @staticmethod
    def _torque_grid(lut, u: np.ndarray) -> np.ndarray:
        """Torque tau_0 + K·u [Nm] produced by u (n_W, A, V, T, M) at every grid point: (n_W, A, V, T)."""
        return lut.tau_0[None, :, None, None] + np.einsum("avm,wavtm->wavt", lut.k, u)

    # ----- validation -----
    def validate(self, lut) -> dict:
        """Metrics on a validation LUT (points not in the training grid, exact targets): saved in
        self.validation_metrics, together with the name of the LUT they refer to."""
        if lut.lut_name == self.lut_name:
            raise ValueError(f"'{lut.lut_name}' is the training LUT: it cannot be used for validation")
        if lut.p != self.p or list(lut.muscles_names) != list(self.muscles_names):
            raise ValueError(f"'{lut.lut_name}' has a different p or different muscles from the training LUT")

        metrics = self.evaluate(lut)
        self.validation_metrics = metrics
        self.validation_lut_name = lut.lut_name
        return metrics

    # ----- get values -----
    def get_activations(self, w, theta, theta_dot, tau) -> np.ndarray:
        """Activations u (N, M) in [0, 1]: same interface as Lut.get_activations (W, angle [deg],
        velocity [deg/s], torque [Nm]; scalars or arrays, broadcast together). Unlike the LUT: any W in [0, 1],
        and no NaN outside the grid (angle and velocity are saturated by the pre-processing)."""
        w, theta, theta_dot, tau = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype="float32"))
                                                         for v in (w, theta, theta_dot, tau)))
        x = {name: values.reshape(-1, 1) for name, values in zip(INPUT_NAMES, (w, theta, theta_dot, tau))}

        if len(w) <= DIRECT_CALL_MAX:
            # few rows (e.g. one step of the controller): direct call, no overhead of predict
            u = keras.ops.convert_to_numpy(self.model(x, training=False))
        else:
            u = self.model.predict(x, batch_size=PREDICT_BATCH, verbose=0)
        return u.astype(float)

    # ----- print -----
    def print_info(self) -> None:
        print(f"Neural network: {self.nn_name}")
        print(f"Trained on LUT: {self.lut_name} (p = {self.p:g})")
        print(f"Structure: {self.num_hidden_layers} hidden layers of {self.hidden_units} neurons, "
              f"hidden function '{self.hidden_function}', output function '{self.output_function}', "
              f"seed {self.seed}")
        print(f"Parameters: {self.num_parameters}")
        print(f"Training: {self.epochs} epochs, batch size {self.batch_size}, "
              f"lr {self.learning_rate:g} -> {self.learning_rate * self.lr_final_factor:g} (cosine decay)")
        print(f"Final loss: {self.history['loss'][-1]:.3e}")
        self._print_metrics("Train metrics", self.train_metrics)
        if self.validation_metrics:
            self._print_metrics(f"Validation metrics ({self.validation_lut_name})", self.validation_metrics)
    @staticmethod
    def _print_metrics(title: str, metrics: dict) -> None:
        # only the scalar metrics: the per-muscle vectors are in the metadata file
        print(f"{title}:")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {name:<20}: {value:.4g}")

    # ----- plots -----
    def plot_structure(self, save_path: str = None, show: bool = True) -> None:
        """Graph of the layers (inputs, pre-processing, hidden layers, u_raw, clip).
        Keras always writes a file: without save_path it goes in a temporary folder.
        Needs pydot and Graphviz installed."""
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            to_file = save_path
        else:
            to_file = os.path.join(tempfile.gettempdir(), f"{self.nn_name}_graph.png")
        image = keras.utils.plot_model(self.model, to_file=to_file, show_shapes=True,
                                       show_layer_names=True, rankdir="LR", dpi=150)
        if save_path is not None:
            print(f"Plot saved in: {save_path}")
        if show and image is not None:
            from IPython.display import display   # only in a notebook
            display(image)

    def plot_loss(self, save_path: str = None, show: bool = True) -> None:
        epochs = np.arange(1, len(self.history["loss"]) + 1)
        plot_cartesian(title=f"{self.nn_name} ({self.num_parameters} param): training loss",
                       x=epochs,
                       y=np.asarray(self.history["loss"]),       # array: a list would be read as many curves
                       x_label="epoch",
                       y_label="loss (MSE)",
                       log_y=True,
                       save_path=save_path,
                       show=show)

    def plot_learning_rate(self, save_path: str = None, show: bool = True) -> None:
        epochs = np.arange(1, len(self.history["learning_rate"]) + 1)
        plot_cartesian(title=f"{self.nn_name}: learning rate",
                       x=epochs,
                       y=np.asarray(self.history["learning_rate"]),
                       x_label="epoch",
                       y_label="learning rate",
                       log_y=True,
                       save_path=save_path,
                       show=show)

    def plot_heatmaps_activations(self, lut, w: float, velocity: float = 0.0,
                                  theta_req: float = None, tau_norm_req: float = None,
                                  save_path: str = None, show: bool = True) -> None:
        """u of the network over (angle, torque_norm), at the grid W and velocity of the LUT nearest to w, velocity."""
        iw, iv, u = self._predict_slice(lut, w, velocity)
        plot_heatmaps(title=rf"{self.nn_name}: u at W = {lut.w[iw]:g}, "
                            rf"$\dot{{\theta}}$ = {lut.theta_dot[iv]:g} deg/s",
                      x=lut.theta,
                      y=lut.tau_norm,
                      z=[u[:, :, i] for i in range(u.shape[-1])],
                      subplot_titles=self._subplot_titles(),
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\tau_{norm}$",
                      colorbar_label="u",
                      vmin=0.0,
                      vmax=1.0,
                      x_req=theta_req,
                      y_req=tau_norm_req,
                      marker_color="red",
                      save_path=save_path,
                      show=show)

    def plot_heatmaps_error(self, lut, w: float, velocity: float = 0.0,
                            theta_req: float = None, tau_norm_req: float = None,
                            save_path: str = None, show: bool = True) -> None:
        """u_NN - u_LUT over (angle, torque_norm): red = network too high, blue = too low."""
        iw, iv, u = self._predict_slice(lut, w, velocity)
        error = u - lut.activations[iw, :, iv, :, :]                     # (A, T, M)
        plot_heatmaps(title=rf"{self.nn_name}: $u_{{NN}} - u_{{LUT}}$ at W = {lut.w[iw]:g}, "
                            rf"$\dot{{\theta}}$ = {lut.theta_dot[iv]:g} deg/s",
                      x=lut.theta,
                      y=lut.tau_norm,
                      z=[error[:, :, i] for i in range(error.shape[-1])],
                      subplot_titles=self._subplot_titles(),
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\tau_{norm}$",
                      colorbar_label="error",
                      cmap="bwr",
                      centered=True,
                      x_req=theta_req,
                      y_req=tau_norm_req,
                      marker_color="black",
                      save_path=save_path,
                      show=show)

    def plot_heatmap_torque_error(self, lut, w: float, velocity: float = 0.0,
                                  theta_req: float = None, tau_norm_req: float = None,
                                  save_path: str = None, show: bool = True) -> None:
        """tau_0 + K·u_NN - tau_ref [Nm] over (angle, torque_norm): red = torque too high, blue = too low."""
        iw, iv, u = self._predict_slice(lut, w, velocity)
        tau_nn = lut.tau_0[:, None] + np.einsum("am,atm->at", lut.k[:, iv, :], u)    # (A, T)
        torque_error = tau_nn - lut.tau[:, iv, :]
        plot_heatmaps(title=rf"{self.nn_name}: torque error at W = {lut.w[iw]:g}, "
                            rf"$\dot{{\theta}}$ = {lut.theta_dot[iv]:g} deg/s",
                      x=lut.theta,
                      y=lut.tau_norm,
                      z=[torque_error],
                      subplot_titles=[r"$\tau_0 + K \cdot u_{NN} - \tau_{ref}$"],
                      x_label=r"$\theta$ [deg]",
                      y_label=r"$\tau_{norm}$",
                      colorbar_label="torque error [Nm]",
                      cmap="bwr",
                      centered=True,
                      x_req=theta_req,
                      y_req=tau_norm_req,
                      marker_color="black",
                      ncols=1,
                      save_path=save_path,
                      show=show)

    def plot_parity(self, lut, w: float = None, velocity: float = None, max_points: int = 20000,
                    seed: int = 0, save_path: str = None, show: bool = True) -> None:
        """u_NN against u_LUT, one panel per muscle: on the whole grid, or only at one W and/or one velocity
        (the grid values nearest to w, velocity). RMSE and R2 on all the selected points; at most max_points
        random points are drawn (always the same ones for the same seed)."""
        u_pred = np.clip(self._predict_grid(lut), 0.0, 1.0)            # (n_W, A, V, T, M)
        u_target = lut.activations

        # selection: one W and/or one velocity of the grid (None = all of them)
        selection = []
        iw, iv = slice(None), slice(None)
        if w is not None:
            iw = int(np.argmin(np.abs(lut.w - w)))
            selection.append(f"W = {lut.w[iw]:g}")
        if velocity is not None:
            iv = int(np.argmin(np.abs(lut.theta_dot - velocity)))
            selection.append(f"velocity = {lut.theta_dot[iv]:g} deg/s")
        n_outputs = u_pred.shape[-1]
        u_pred = u_pred[iw, :, iv].reshape(-1, n_outputs)             # (N, M)
        u_target = u_target[iw, :, iv].reshape(-1, n_outputs)

        # metrics on all the selected points
        metrics = compute_metrics(u_target, u_pred)
        texts = [f"RMSE {rmse:.3f}   R² {r2:.4f}"
                 for rmse, r2 in zip(metrics["rmse_per_muscle"], metrics["r2_per_muscle"])]

        # points drawn: a random subset, the same for the same seed
        n_points = len(u_target)
        rng = np.random.default_rng(seed)
        drawn = rng.choice(n_points, size=min(max_points, n_points), replace=False)

        plot_parity(title=f"{self.nn_name} ({self.num_parameters} param): parity, "
                    f"{', '.join(selection) if selection else 'whole grid'}",
            x=[[u_target[drawn, i] for i in range(n_outputs)]],      # one series: this network
            y=[[u_pred[drawn, i] for i in range(n_outputs)]],
            labels=[f"{self.nn_name} ({self.num_parameters} param)"],
            subplot_titles=self._subplot_titles(),
            texts=[texts],
            x_label="u LUT",
            y_label="u NN",
            save_path=save_path,
            show=show)
        
    # ----- utils for the plots -----
    def _predict_slice(self, lut, w: float, velocity: float):
        """u of the network on one slice of the LUT grid: all angles and torques, at the grid W and velocity
        nearest to w, velocity. Returns iw, iv and u (A, T, M)."""
        iw = int(np.argmin(np.abs(lut.w - w)))
        iv = int(np.argmin(np.abs(lut.theta_dot - velocity)))
        n_angles, n_torques = len(lut.theta), len(lut.tau_norm)

        theta = np.repeat(lut.theta, n_torques)          # angle of every point of the slice, angle by angle
        tau = lut.tau[:, iv, :].ravel()                  # torque [Nm] of the same points, same order
        u = self.get_activations(lut.w[iw], theta, lut.theta_dot[iv], tau)
        return iw, iv, u.reshape(n_angles, n_torques, -1)

    def _subplot_titles(self) -> list:
        return [f"u({i + 1}): {name}" for i, name in enumerate(self.muscles_names)]

    # ----- Utils -----
    def _save_predictions_nc(self, lut, path_nc: str) -> None:
        """u_raw (network, before the clip) and u_target (LUT) on the grid of the training LUT, plus what is
        needed to plot in Nm and to compute the torque error without the LUT. Same coordinates as the LUT file."""
        if lut.lut_name != self.lut_name:
            raise ValueError(f"The network was trained on '{self.lut_name}', not on '{lut.lut_name}'")

        u_raw = self._predict_grid(lut)
        grid_dims = ("W", "angle", "velocity", "torque_norm", "act")
        coords = {"W": lut.w, "angle": lut.theta, "velocity": lut.theta_dot, "torque_norm": lut.tau_norm,
                  "act": np.arange(1, u_raw.shape[-1] + 1)}
        data_vars = {
            "u_raw": (grid_dims, u_raw.astype("float32")),
            "u_target": (grid_dims, lut.activations.astype("float32")),   # act of the LUT (act = u, static)
            "torque": (("angle", "velocity", "torque_norm"), lut.tau),     # torque_ref [Nm] of every grid point
            "tau_0": (("angle",), lut.tau_0),
            "K": (("angle", "velocity", "act"), lut.k),
        }
        attrs = {"nn_name": self.nn_name, "lut_name": self.lut_name, "p": self.p,
                 "muscles_names": ",".join(self.muscles_names)}
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # temporary file + replace: a file on disk is always complete
        temp_file = path_nc + ".tmp"
        ds.to_netcdf(temp_file)
        os.replace(temp_file, path_nc)

    def _predict_grid(self, lut) -> np.ndarray:
        """u_raw (before the clip) at every grid point of a LUT: (n_W, A, V, T, M)."""
        raw_model = keras.Model(self.model.input, self.model.get_layer("u_raw").output)
        u_raw = raw_model.predict(grid_inputs(lut), batch_size=PREDICT_BATCH, verbose=0)
        return u_raw.reshape(lut.activations.shape)
