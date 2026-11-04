import json
import os

import keras
import numpy as np
import pandas as pd

from src import nn_plots
from src.lut import LUT
from src.lut_preprocessing import LUT_preprocessing
from src.nn_metrics import N_OUTPUTS, compute_metrics
from src.nn_validator import torque_to_norm

ACTIVATION_MAP = {
    "relu": keras.activations.relu,
    "gelu": keras.activations.gelu,
    "tanh": keras.activations.tanh,
    "silu": keras.activations.silu,
    "sigmoid": keras.activations.sigmoid
    }

OUTPUT_ACTIVATION_MAP = {
    "linear": keras.activations.linear,  # u_raw = W·h + b
    "sigmoid": keras.activations.sigmoid
    }

class NN_model_from_LUT:

    def __init__(self, model_name: str, this_LUT: LUT):
        self.model_name = model_name
        self.this_LUT = this_LUT

        # structure
        self.num_hidden_layers = None
        self.hidden_units = None
        self.num_parameters = None
        self.activation_name = None
        self.output_activation = None
        self.seed = None

        # models (shared weights)
        self.train_model = None
        self.this_model = None

        # data split (filled by prepare_data)
        self.data = None

        # training
        self.history = None
        self.train_metrics = None
        self.validation_metrics = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.lr_factor = None
        self.lr_patience = None
        self.min_lr = None

    @property
    def label(self) -> str:
        """Label used in legends and titles of the plots."""
        return f"{self.model_name} ({self.num_parameters} param)"

    def define_structure(self,
                         num_hidden_layers: int,
                         hidden_units: int,
                         activation: str = "relu",
                         output_activation: str = "linear",
                         seed: int = 42,
                         show_table: bool = False,
                         save_path_plots: str = None
                         ) -> None:
        # errors wrong activation spell
        if activation not in ACTIVATION_MAP:
            raise ValueError(f"activation must be one of {list(ACTIVATION_MAP)}, got '{activation}'")
        if output_activation not in OUTPUT_ACTIVATION_MAP:
            raise ValueError(f"output_activation must be one of {list(OUTPUT_ACTIVATION_MAP)}, got '{output_activation}'")

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.activation_name = activation.lower()
        self.output_activation = output_activation.lower()
        self.seed = seed
        keras.utils.set_random_seed(seed)

        # inputs in physical units
        W_in = keras.Input(shape=(1,), name="W")
        angle_in = keras.Input(shape=(1,), name="angle")
        torque_in = keras.Input(shape=(1,), name="torque")
        inputs = {"W": W_in, "angle": angle_in, "torque": torque_in}

        # pre-processing: constants read once from the LUT and saved inside the model
        preprocessing = LUT_preprocessing.from_lut(self.this_LUT, name="preprocessing")
        features, _ = preprocessing([W_in, angle_in, torque_in])

        # hidden layers
        act_fn = ACTIVATION_MAP[self.activation_name]
        h = features
        for i in range(num_hidden_layers):
            h = keras.layers.Dense(hidden_units, activation=act_fn, name=f"hidden_{i}")(h)

        # output layer: the network is trained on u_raw
        out_fn = OUTPUT_ACTIVATION_MAP[self.output_activation]
        u_raw = keras.layers.Dense(N_OUTPUTS, activation=out_fn, name="u_raw")(h)
        u = keras.layers.ReLU(max_value=1.0, name="u")(u_raw)  # = clip(u_raw, 0, 1)

        self.train_model = keras.Model(inputs, u_raw, name=f"{self.model_name}_train")
        self.this_model = keras.Model(inputs, u, name=self.model_name)

        self.num_parameters = self.this_model.count_params()
        print(f"Architecture of the NN: {num_hidden_layers} layers of {hidden_units} neurons, "
              f"activation '{activation}', output '{output_activation}' ({self.num_parameters} param).")

        if show_table or (save_path_plots is not None):
            self.plot_structure(show=show_table, save_path=save_path_plots)

    def train(self,
              epochs: int,
              batch_size: int,
              lr: float,
              loss: str = "mse",
              patience: int = 50,
              min_delta: float = 0,
              lr_factor: float = 0.5,
              lr_patience: int = 20,
              min_lr: float = 1e-6,
              verbose: int = 2,
              show_plots: bool = False,
              save_path_plots: str = None) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        # load data (physical values -> the model normalizes them itself)
        x, y = self._get_xy(self.this_LUT)
        print(f"Number of rows in the dataset: {len(y)}")
        print(f"W values: {np.unique(self.this_LUT.W)}")
        print(f"Training for at most {self.epochs} epochs...\n")

        """
        # loss on u_raw (no clip): the gradient never vanishes
        self.train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss,
            metrics=["mae"],
        )

        # early stop: stop after `patience` epochs without improvement, keep the best weights
        early_stop = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=True,
        )

        # lr scheduler: halve the lr after `lr_patience` epochs without improvement
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr,
            min_delta=min_delta,
            verbose=verbose,
        )
        
        history = self.train_model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,          # rows are sorted by W: mix them in every epoch
            callbacks=[reduce_lr, early_stop],
            verbose=verbose,
        )
        """

        steps_per_epoch = int(np.ceil(len(y) / batch_size))
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-3,                 # lr finale = lr * 1e-3 (es. 1e-3 -> 1e-6)
        )

        self.train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=schedule),
            loss=loss,
            metrics=["mae"],
            #jit_compile=False,
        )

        history = self.train_model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose,
        )
        # FINE TENTATIVO 2


        self.history = {k: [float(v) for v in values] for k, values in history.history.items()}
        print(f"\nTraining finished after {len(self.history['loss'])} epochs "
              f"(best loss {min(self.history['loss']):.3e}).")

        # metrics on the final model (with the clip), the one that will be used
        u_pred = keras.ops.convert_to_numpy(self.this_model(x, training=False))
        self.train_metrics = compute_metrics(y, u_pred)
        m = self.train_metrics
        print(f"MSE {m['mse']:.3e} | RMSE {m['rmse']:.4f} | max |err| {m['max_error']:.4f} | "
              f"R2 {m['r2']:.4f} | FIT {m['fit_pct']:.2f} %")

        if show_plots or (save_path_plots is not None):
            self.plot_history(show=show_plots, save_path=save_path_plots)
            self.plot_predictions(show=show_plots, save_path=save_path_plots)

    def _get_xy(self, lut: LUT):
        """Inputs (dict of physical values) and targets of a loaded LUT, in the format of Keras."""
        if lut.activations is None:
            raise RuntimeError("LUT not loaded: call load_data() first")
        x = {
            "W": lut.W.reshape(-1, 1).astype("float32"),
            "angle": lut.angle.reshape(-1, 1).astype("float32"),
            "torque": lut.torque.reshape(-1, 1).astype("float32"),
        }
        y = lut.activations.astype("float32")
        return x, y

    def export(self, path_export: str) -> None:
        os.makedirs(path_export, exist_ok=True)
        base = os.path.join(path_export, self.model_name)

        # 1. model (network + pre-processing + clip): the only file the controller needs
        self.this_model.save(f"{base}.keras")
        print(f"Model Keras saved in: {base}.keras")

        # 2. history: one row per epoch
        self.history_table().to_csv(f"{base}_history.csv", index=False)
        print(f"History saved in: {base}_history.csv")

        # 3. predictions: one row per LUT point (u_target vs u_NN, for MATLAB)
        self.predictions_table().to_csv(f"{base}_predictions.csv", index=False)
        print(f"Predictions saved in: {base}_predictions.csv")

        # 4. metadata: everything needed to know how the model was obtained
        metadata = {
            "model_name": self.model_name,
            #"lut_file": os.path.abspath(self.this_LUT.nc_file_path),
            "W_values": np.unique(self.this_LUT.W).tolist(),
            "num_rows": int(len(self.this_LUT.W)),
            "hyperparameters": {
                "num_parameters": self.num_parameters,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_units": self.hidden_units,
                "activation": self.activation_name,
                "output_activation": self.output_activation,
                "seed": self.seed,
                "epochs": self.epochs,
                "epochs_run": len(self.history["loss"]),
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "lr_factor": self.lr_factor,
                "lr_patience": self.lr_patience,
                "min_lr": self.min_lr,
                "final_learning_rate": self.history.get("learning_rate", [None])[-1],
            },
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "keras_version": keras.__version__,
            "keras_backend": keras.backend.backend(),
            "files": {
                "model": f"{self.model_name}.keras",
                "history": f"{self.model_name}_history.csv",
                "predictions": f"{self.model_name}_predictions.csv",
            },
        }
        with open(f"{base}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata and metrics saved in: {base}_metadata.json")

    def history_table(self) -> pd.DataFrame:
        """One row per epoch: epoch, loss, mae (same format as the exported CSV)."""
        if self.history is None:
            raise RuntimeError("No training history: call train() first")
        df = pd.DataFrame(self.history)
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
        return df

    def predictions_table(self, lut: LUT = None) -> pd.DataFrame:
        """One row per LUT point: inputs, u_target, u_raw, u_pred (with the clip) and
        error = u_pred - u_target."""
        if lut is None:
            lut = self.this_LUT
        x, y = self._get_xy(lut)
        u_pred = keras.ops.convert_to_numpy(self.this_model(x, training=False))
        u_raw = keras.ops.convert_to_numpy(self.train_model(x, training=False))

        df = pd.DataFrame({
            "W": lut.W,
            "angle": lut.angle,
            "torque": lut.torque,
            "torque_norm": lut.torque_norm,
        })
        for i in range(N_OUTPUTS):
            df[f"u_target_{i}"] = y[:, i]
        for i in range(N_OUTPUTS):
            df[f"u_raw_{i}"] = u_raw[:, i]
        for i in range(N_OUTPUTS):
            df[f"u_pred_{i}"] = u_pred[:, i]
        for i in range(N_OUTPUTS):
            df[f"error_{i}"] = u_pred[:, i] - y[:, i]
        return df

    def predict(self, W, angle, torque) -> np.ndarray:
        """u(0...5) for physical inputs (scalars or arrays of the same length). Returns (N, 6)."""
        if self.this_model is None:
            raise RuntimeError("Model not built yet: call define_structure() first")
        W, angle, torque = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype="float32"))
                                                  for v in (W, angle, torque)))
        x = {"W": W.reshape(-1, 1), "angle": angle.reshape(-1, 1), "torque": torque.reshape(-1, 1)}
        return keras.ops.convert_to_numpy(self.this_model(x, training=False))

    # ----------------------------------------------------------------- plots

    @staticmethod
    def _W_tag(considered_W) -> str:
        return "all" if considered_W is None else f"W_{considered_W:g}"

    def plot_history(self, show: bool = False, save_path: str = None) -> None:
        nn_plots.plot_history({self.label: self.history_table()}, metric="loss",
                              title=f"{self.model_name}: training loss",
                              save_path=save_path, file_name=f"{self.model_name}_loss", show=show)

    def plot_predictions(self, considered_W: float = None, angle_range: tuple = None,
                         torque_norm_range: tuple = None, lut: LUT = None,
                         show: bool = False, save_path: str = None) -> None:
        """Series, error, parity and error histogram of u_pred vs u_target.
        Files: {model}_predictions_{all | W_x}_{series | error | parity | hist}.png"""
        df = self.predictions_table(lut)
        if considered_W is not None and not np.isclose(df["W"], considered_W).any():
            raise ValueError(f"W={considered_W} not found. Available: {np.unique(df['W'])}")

        frames = {self.label: df}
        filters = (considered_W, angle_range, torque_norm_range)
        prefix = f"{self.model_name}_predictions_{self._W_tag(considered_W)}"
        common = dict(save_path=save_path, show=show)

        nn_plots.plot_series(frames, *filters, file_name=f"{prefix}_series", **common)
        nn_plots.plot_series(frames, *filters, error=True, file_name=f"{prefix}_error", **common)
        nn_plots.plot_parity(frames, *filters, file_name=f"{prefix}_parity", **common)
        nn_plots.plot_error_hist(frames, *filters, file_name=f"{prefix}_hist", **common)

    def plot_raw_vs_pred(self, considered_W: float = None, angle_range: tuple = None,
                         torque_norm_range: tuple = None, lut: LUT = None,
                         show: bool = False, save_path: str = None) -> None:
        """u_raw (before the clip) and u_pred. File: {model}_predictions_{all | W_x}_u_raw_u_pred.png"""
        nn_plots.plot_raw_vs_pred(
            {self.label: self.predictions_table(lut)}, considered_W, angle_range, torque_norm_range,
            file_name=f"{self.model_name}_predictions_{self._W_tag(considered_W)}_u_raw_u_pred",
            save_path=save_path, show=show)

    def plot_structure(self, show: bool = False, save_path: str = None) -> None:
        if self.this_model is None:
            raise RuntimeError("Model not built yet: call define_structure() first")

        if show:
            self.this_model.summary()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            # text summary: always available
            summary_path = os.path.join(save_path, f"{self.model_name}_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                self.this_model.summary(print_fn=lambda line, **kwargs: f.write(line + "\n"))
            print(f"Architecture summary saved in: {summary_path}")

            # graph
            file_path = os.path.join(save_path, f"{self.model_name}_graph.png")
            try:
                keras.utils.plot_model(self.this_model, to_file=file_path, show_shapes=True,
                                       show_layer_names=True, rankdir="LR", dpi=150)
                print(f"Architecture plot saved in: {file_path}")
            except ImportError as e:
                # plot_model needs pydot and graphviz
                print(f"Architecture plot not saved: {e}")

    def plot_heatmaps(self, considered_W: float, angle_req: float = None, torque_req: float = None,
                      lut: LUT = None, show: bool = True, save_path: str = None) -> dict:
        """Heatmaps of u_pred(angle, torque_norm) for one W. torque_req in Nm."""
        return self._draw_nn_heatmaps("u_pred", considered_W, angle_req, torque_req, lut, show, save_path)

    def plot_error_heatmaps(self, considered_W: float, angle_req: float = None, torque_req: float = None,
                            lut: LUT = None, show: bool = True, save_path: str = None) -> dict:
        """Heatmaps of error = u_pred - u_target for one W (red: prediction too high). torque_req in Nm."""
        return self._draw_nn_heatmaps("error", considered_W, angle_req, torque_req, lut, show, save_path)

    def _draw_nn_heatmaps(self, value: str, considered_W: float, angle_req, torque_req,
                          lut: LUT, show: bool, save_path: str) -> dict:
        if lut is None:
            lut = self.this_LUT
        df = self.predictions_table(lut)
        if not np.isclose(df["W"], considered_W).any():
            raise ValueError(f"W={considered_W} not found. Available: {np.unique(lut.W)}")

        # requested point: torque in Nm -> torque_norm at that angle (as the pre-processing does)
        tau_req = None
        if angle_req is not None and torque_req is not None:
            tau_req, t_min, t_max = torque_to_norm(lut, angle_req, torque_req)
            if not 0.0 <= tau_req <= 1.0:
                print(f"Torque {torque_req} Nm is outside [{t_min:.2f}, {t_max:.2f}] Nm at {angle_req} deg: "
                      f"no marker")
                tau_req = None
            else:
                # the heatmap shows the nearest grid point; the network gives the exact value
                u_exact = self.predict(considered_W, angle_req, torque_req)[0]
                print(f"NN prediction at exactly (W={considered_W:g}, angle={angle_req}, torque={torque_req} Nm, "
                      f"torque_norm={tau_req:.3f}): " + ", ".join(f"{v:.4f}" for v in u_exact))

        return nn_plots.plot_heatmaps(
            {self.model_name: df}, value=value, W=considered_W,
            angle_req=angle_req, tau_req=tau_req,
            save_path=save_path, show=show,
            file_name=f"{self.model_name}_{value}_heatmaps_W_{considered_W:g}")