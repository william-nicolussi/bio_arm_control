import keras
import numpy as np

from src.lut.lut import Lut
from src.nn.nn import NeuralNetwork, INPUT_NAMES, grid_inputs
from src.nn.nn_preprocessing import NnPreprocessing

HIDDEN_FUNCTION = {
    "relu": keras.activations.relu,
    "gelu": keras.activations.gelu,
    "tanh": keras.activations.tanh,
    "silu": keras.activations.silu,
    "sigmoid": keras.activations.sigmoid,
}

OUTPUT_FUNCTION = {
    "linear": keras.activations.linear,  # u_raw = W·h + b
}


class LearningRateLogger(keras.callbacks.Callback):
    """Learning rate at the end of every epoch: with a schedule Keras does not log it."""

    def __init__(self):
        super().__init__()
        self.values = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate          # with a schedule: value at the current step
        self.values.append(float(keras.ops.convert_to_numpy(lr)))


class NnGenerator:
    def __init__(self, lut: Lut):
        self.lut = lut
        self.n_outputs = lut.activations.shape[-1]      # one u per muscle

        # training data, prepared once for all the networks of this generator
        self.x = grid_inputs(lut)                                      # dict of (N, 1), physical units
        self.y = lut.activations.reshape(-1, self.n_outputs).astype("float32")   # target u <- act of the LUT
                                                                       # (static hypothesis: act = u)
        # structure (set by define_structure_nn)
        self.num_hidden_layers = None
        self.hidden_units = None
        self.hidden_function = None
        self.output_function = None
        self.seed = None

        # models with shared weights (set by define_structure_nn, used once by train_nn)
        self.train_model = None     # output u_raw: trained with the loss
        self.model = None           # output u = clip(u_raw, 0, 1): goes in the NeuralNetwork

        # last network produced
        self.nn = None

    # ----- define -----
    def define_structure_nn(self,
                            num_hidden_layers: int,
                            hidden_units: int,
                            hidden_function: str = "relu",
                            output_function: str = "linear",
                            seed: int = 42) -> None:
        hidden_function = hidden_function.lower()
        output_function = output_function.lower()
        if hidden_function not in HIDDEN_FUNCTION:
            raise ValueError(f"hidden_function must be one of {list(HIDDEN_FUNCTION)}, got '{hidden_function}'")
        if output_function not in OUTPUT_FUNCTION:
            raise ValueError(f"output_function must be one of {list(OUTPUT_FUNCTION)}, got '{output_function}'")

        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.hidden_function = hidden_function
        self.output_function = output_function
        self.seed = seed
        keras.utils.set_random_seed(seed)       # same seed -> same initial weights

        # inputs in physical units
        inputs = {name: keras.Input(shape=(1,), name=name) for name in INPUT_NAMES}

        # pre-processing: constants read once from the LUT and saved inside the model
        preprocessing = NnPreprocessing.from_lut(self.lut, name="preprocessing")
        features, _ = preprocessing([inputs[name] for name in INPUT_NAMES])

        # hidden layers ("activation" is the name Keras gives to the neuron function)
        h = features
        for i in range(num_hidden_layers):
            h = keras.layers.Dense(hidden_units, activation=HIDDEN_FUNCTION[hidden_function],
                                   name=f"hidden_{i}")(h)

        # output layer: the network is trained on u_raw, the controller uses u
        u_raw = keras.layers.Dense(self.n_outputs, activation=OUTPUT_FUNCTION[output_function],
                                   name="u_raw")(h)
        u = keras.layers.ReLU(max_value=1.0, name="u")(u_raw)   # = clip(u_raw, 0, 1)

        self.train_model = keras.Model(inputs, u_raw, name="train_model")
        self.model = keras.Model(inputs, u, name="model")

        print(f"Structure: {num_hidden_layers} hidden layers of {hidden_units} neurons, "
              f"hidden function '{hidden_function}', output function '{output_function}' "
              f"({self.model.count_params()} param).")

    # ----- train -----
    def train_nn(self,
                 nn_name: str,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 lr_final_factor: float = 1e-3,
                 verbose: int = 2,
                 print_result: bool = True) -> None:
        """Train the structure defined by define_structure_nn with MSE on u_raw and a cosine decay of the
        learning rate (learning_rate at the first step, learning_rate * lr_final_factor at the last one).
        The trained network goes in self.nn. A structure is trained once: call define_structure_nn again
        for the next network (same seed -> same initial weights)."""
        if self.train_model is None:
            raise RuntimeError("No structure to train: call define_structure_nn() first")

        print(f"Training '{nn_name}' on {len(self.y)} rows for {epochs} epochs...\n")

        steps_per_epoch = int(np.ceil(len(self.y) / batch_size))
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=epochs * steps_per_epoch,
            alpha=lr_final_factor,
        )

        # loss on u_raw (no clip): the gradient never vanishes
        self.train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=schedule),
            loss="mse",
            metrics=["mae"],
        )

        lr_logger = LearningRateLogger()
        history = self.train_model.fit(
            self.x, self.y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,              # rows are sorted by W: mix them in every epoch
            callbacks=[lr_logger],
            verbose=verbose,
        )
        history = {k: [float(v) for v in values] for k, values in history.history.items()}
        history["learning_rate"] = lr_logger.values

        self.nn = NeuralNetwork(nn_name=nn_name,
                                model=self.model,
                                lut_name=self.lut.lut_name,
                                p=self.lut.p,
                                muscles_names=list(self.lut.muscles_names),
                                num_hidden_layers=self.num_hidden_layers,
                                hidden_units=self.hidden_units,
                                hidden_function=self.hidden_function,
                                output_function=self.output_function,
                                seed=self.seed,
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                lr_final_factor=lr_final_factor,
                                history=history)
        self.nn.train_metrics = self.nn.evaluate(self.lut)

        # the trained weights now belong to self.nn: the next network needs a new structure
        self.train_model = None
        self.model = None

        if print_result:
            m = self.nn.train_metrics
            print(f"\nTraining finished: final loss {history['loss'][-1]:.3e}")
            print(f"RMSE {m['rmse']:.4f} | max |err| {m['max_error']:.4f} | R2 {m['r2']:.4f} | "
                f"FIT {m['fit_pct']:.2f} %")
            print(f"Torque error: mean {m['torque_err_mean_Nm']:.3f} Nm | p95 {m['torque_err_p95_Nm']:.3f} Nm | "
                f"max {m['torque_err_max_Nm']:.3f} Nm")