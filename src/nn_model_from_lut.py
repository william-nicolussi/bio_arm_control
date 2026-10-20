import json
import os

import keras
import numpy as np
import pandas as pd

from nnodely import (
    GELU,
    Clamp,
    Concatenate,
    DataLoader,
    Input,
    Linear,
    Modely,
    Output,
    ReLU,
    Sigmoid,
    Swish,
    Tanh,
    set_seed,
)

from src.draw_6x2_heatmaps import draw_heatmaps
from src.lut import LUT

ACTIVATION_MAP = {
    "relu": ReLU,
    "gelu": GELU,
    "tanh": Tanh,
    "swish": Swish,
    "sigmoid": Sigmoid,
}

class NN_model_from_LUT:
  
    def __init__(self, model_name: str, this_LUT: LUT):
        self.model_name = model_name
        self.this_LUT = this_LUT

        self.num_hidden_layers = None
        self.hidden_units = None
        self.num_parameters = None
        self.activation_name = None

        self.this_model = None
        self.history = None
        self.validation_metrics = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.seed = None

    def define_structure(self, 
                         num_hidden_layers: int, 
                         hidden_units: int, 
                         activation: str = "relu",
                         show_table: bool = False,
                         save_path_plots: str = None
                         ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.hidden_units = hidden_units
        self.activation_name = activation.lower()

        # input for Modely object
        angle = Input("angle", dim=1)
        torque_norm = Input("torque_norm", dim=1)

        angle_norm = (angle.last() - self.this_LUT.angle_min) / (self.this_LUT.angle_max - self.this_LUT.angle_min)

        # input layer x_in
        x_in = Concatenate()([angle_norm, torque_norm.last()])

        # hidden layers
        act_class = ACTIVATION_MAP.get(self.activation_name, ReLU)
        h = x_in
        for _ in range(self.num_hidden_layers):
            h = act_class()(Linear(out_features=self.hidden_units)(h))

        # output layer u_out
        #u_out = Linear(out_features=6)(h)
        u_raw = Linear(out_features=6)(h)
        u_out = Sigmoid()(u_raw) # <- or Clamp

        # output of Modely object
        u_pred = Output("u_pred", u_out)

        # define the model
        self.this_model = Modely(
            name=self.model_name,
            inputs=[angle, torque_norm],
            outputs=[u_pred]
        )

        # define the target function (u_target from LUT)
        u_target = Input("u_target", dim=6)
        self.this_model.minimize(
            name=f"{self.model_name}_loss",
            source=u_pred,
            target=u_target.last(),
            loss="mse" # default
        )

        # all information to build the model
        self.this_model.build()

        self.num_parameters = self.this_model.model.count_params()
        print(f"Architecture of the NN: {self.num_hidden_layers} layers of {self.hidden_units} neurons each ({self.num_parameters} param).")

        # plots and save only if user wants them
        if show_table or (save_path_plots is not None):
            self.plot_structure(show=show_table, save_path=save_path_plots)

    def plot_structure(self, show: bool = False, save_path: str = None) -> None:
        if show:
            self.this_model.summary()

        if save_path is not None:
            self.this_model.plot(f"{save_path}/graph_nn.png")
            self.this_model.export_html(out_dir=save_path, filename="nn_architecture.html")
            print(f"Architecture plots saved in: {save_path}")

    def train(self, 
              epochs: int, 
              batch_size: int, 
              lr: float, 
              seed: int = 42,
              show_plots: bool = False,
              save_path_plots: str = None) -> None:
        
        self.seed = seed
        set_seed(self.seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = lr

        # load data
        full_dataset = DataLoader(self.this_model, source=self.this_LUT.lut_data)
        print(f"Number of rows in the dataset: {len(self.this_LUT.lut_data['angle'])}")
        print(f"Training for {self.epochs} epochs...\n")

        # training with EARLY STOP -> 15 epochs withou improvement it stops
        self.history = self.this_model.train(
            full_dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            shuffle=True, # rearrange data (avoid catastrophic forgetting)
            early_stopping="loss", # go on only if error gos down
            early_stopping_kwargs={"patience": 15, "restore_best_weights": True},
            printer="legacy" 
        )
        print("\nTraining finished!")

        # plot and save (if user wants)
        name_minimizer = f"{self.model_name}_loss"
        result = self.this_model.validate(
            full_dataset,
            history=self.history,
            out_dir=save_path_plots,
            show=show_plots,
            verbose=show_plots,
        )

        # save metrics MSE, Fit, R2...
        self.validation_metrics = result[name_minimizer].metrics
        
    def export(self, path_export) -> None:
        os.makedirs(path_export, exist_ok=True)

        self.this_model.export_keras(path_export, filename=self.model_name)
        print(f"\nModel Keras saved in: {path_export}/{self.model_name}.keras")

        self.this_model.save(f"{path_export}/{self.model_name}")
        print(f"Model nnodely saved in: {path_export}/{self.model_name}/")

        # fill for json
        metadata = {
            "model_name": self.model_name,
            "hyperparameters": {
                "num_parameters": self.num_parameters,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_units": self.hidden_units,
                "activation": self.activation_name,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
            "validation_metrics": self.validation_metrics,
            "history": self.history,
        }

        # export json
        metadata_path = f"{path_export}/{self.model_name}_metadata.json"
        open(metadata_path, "w").write(
            json.dumps(metadata, default=float, indent=4)
        )

        print(f"Metadata, metrics and history saved in: {metadata_path}")

    def plot_heatmaps(self, angle_req: float = None, tau_req: float = None, save_path: str = None) -> None:
        # create vector using the NN
        full_dataset = DataLoader(self.this_model, source=self.this_LUT.lut_data)
        raw_pred = self.this_model(full_dataset.as_dict())["u_pred"]
        full_predictions = np.squeeze(keras.ops.convert_to_numpy(raw_pred))

        # create the dataframe populated using the NN
        df_pred = pd.DataFrame({
            'angle': self.this_LUT.lut_data['angle'],
            'torque_norm': self.this_LUT.lut_data['torque_norm']
        })
        for i in range(6):
            df_pred[i] = full_predictions[:, i]

        if save_path is not None:
            full_save_path = f"{save_path}/{self.model_name}_heatmaps.png"
        else:
            full_save_path = None

        draw_heatmaps(
            df=df_pred,
            angle_min=self.this_LUT.angle_min,
            angle_max=self.this_LUT.angle_max,
            angle_req=angle_req,
            tau_req=tau_req,
            cmap='viridis',
            title_prefix='u_pred',
            marker_color='red',
            center_zero=False,
            save_path=full_save_path
        )

    def plot_error_heatmaps(self, angle_req: float = None, 
                            tau_req: float = None,
                            save_path: str = None) -> None:
        # create vector using the NN
        full_dataset = DataLoader(self.this_model, source=self.this_LUT.lut_data)
        raw_pred = self.this_model(full_dataset.as_dict())["u_pred"]
        full_predictions = np.squeeze(keras.ops.convert_to_numpy(raw_pred))
        
        #full_predictions = self.this_model.predict(full_dataset)

        # errors = u_target - u_NN
        target_activations = self.this_LUT.lut_data["u_target"]
        errors = target_activations - full_predictions

        # temporary dataframe
        df_error = pd.DataFrame({
            'angle': self.this_LUT.lut_data['angle'],
            'torque_norm': self.this_LUT.lut_data['torque_norm']
        })
        for i in range(6):
            df_error[i] = errors[:, i]

        # procedure to download the heatmaps
        if save_path is not None:
            full_save_path = f"{save_path}/{self.model_name}_error_heatmaps.png"
        else:
            full_save_path = None

        draw_heatmaps(
            df=df_error,
            angle_min=self.this_LUT.angle_min,
            angle_max=self.this_LUT.angle_max,
            angle_req=angle_req,
            tau_req=tau_req,
            cmap='bwr', # Blue (<0); White (=0); Red (>0)
            title_prefix='Error u',
            marker_color='black',
            center_zero=True, # 0 is the mean value
            save_path=full_save_path
        )