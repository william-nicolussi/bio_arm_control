import os
import json
import matplotlib.pyplot as plt
import numpy as np
import keras
import nnodely
import pandas as pd
from IPython.display import display
from src.lut import LUT

def return_all_json_path_in_this_folder(folder_path: str) -> list:
    json_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("metadata.json"):
                full_path = os.path.join(root, file)
                normalized_path = full_path.replace('\\', '/') # Windows accepts only '/'
                json_paths.append(normalized_path)   
    return json_paths

def return_all_keras_path_in_this_folder(folder_path: str) -> list:
    keras_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".keras"):
                full_path = os.path.join(root, file)
                normalized_path = full_path.replace('\\', '/') # Windows accepts only '/'
                keras_paths.append(normalized_path)   
    return keras_paths

def print_all_path_in_list(models_to_visualize: list) -> None:
    print("[")
    for model in models_to_visualize:
        print(f"\t'{model}',")
    print("]")


class NN_model_json:

    def __init__(self, json_path: str):
        self.json_path = json_path

        self.model_name = None
        self.hyperparameters = {}
        self.validation_metrics = {}
        self.history = {}

        self.load_json(self.json_path)
        
    def load_json(self, json_path: str) -> None:
        if os.path.exists(json_path):

            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # upload data from json
            self.model_name = json_data.get("model_name")
            self.hyperparameters = json_data.get("hyperparameters", {})
            self.validation_metrics = json_data.get("validation_metrics", {})
            self.history = json_data.get("history", {})
        else:
            raise FileNotFoundError(f"File JSON not found: {json_path}")

        
class NN_model_keras:

    def __init__(self, keras_path: str, this_lut: LUT):
        self.keras_path = keras_path
        self.model_name = os.path.basename(keras_path).replace('.keras', '')
        self.model = None
        self.load_model(self.keras_path)
        self.this_lut = this_lut

    def load_model(self, keras_path: str):
        self.model = keras.models.load_model(keras_path)
        if '=' in self.model.name:
            self.model.name = self.model.name.replace('=', '_')
        for layer in self.model.layers:
            if hasattr(layer, 'name') and '=' in layer.name:
                layer.name = layer.name.replace('=', '_')

    def get_predictions(self) -> np.ndarray:            
        x_input = {
            "angle": self.this_lut.lut_data["angle"].reshape(-1, 1), # dim=1 in nnodely
            "torque_norm": self.this_lut.lut_data["torque_norm"].reshape(-1, 1)
        }
        #u_pred = self.model.predict(x_input)#, verbose=0)
        # return u_pred
        raw_pred = self.model.predict(x_input)
        if isinstance(raw_pred, dict):
            return raw_pred["u_pred"]
        else:
            return raw_pred
        


class NN_models_visualizer:

    def __init__(self):
        self.list_models_json = []
        self.list_models_keras = []

    def add_model_json(self, new_model: NN_model_json) -> None:
        self.list_models_json.append(new_model)

    def add_model_keras(self, new_model: NN_model_keras) -> None:
        self.list_models_keras.append(new_model)

    def plot_loss(self, save_path: str = None) -> None:
        plt.figure(figsize=(10, 8))
        for model in self.list_models_json:
            loss_values = model.history.get("loss", [])
            plt.plot(loss_values, label=f"{model.model_name} ({model.hyperparameters['num_parameters']} param)")

        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        plt.grid(True)

        if save_path is not None:
            full_save_path = f"{save_path}/loss_comparison.png"
            folder = os.path.dirname(full_save_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            plt.savefig(full_save_path)
            print(f"Graph loss saved in: {full_save_path}")

        plt.show()

    def print_table(self, metrics_in_table: list = None) -> None:

        # if method called empty, add all validation_metrics
        if not metrics_in_table:
            metrics_set = set()
            for model in self.list_models_json:
                metrics_set.update(model.validation_metrics.keys())
            metrics_in_table = list(metrics_set)

        # populate table
        data = []
        for model in self.list_models_json:
            row_data = {"Model Name": model.model_name}
            for metric in metrics_in_table:
                row_data[metric] = model.validation_metrics.get(metric, None)
            data.append(row_data)

        df = pd.DataFrame(data) # dataframe for pandas
        formatted_df = df.style.format(precision=4, na_rep="N/A") #4->dig after ','; "N/A"->if data unavailable
        
        display(formatted_df)

    def plot_u_req_u_target(self, save_path: str = None) -> None:
        lut_reference = self.list_models_keras[0].this_lut # use LUT of the first model
                
        u_target = lut_reference.lut_data["u_target"] # extract u_target from LUT
        
        fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True) # 6 subplots
        fig.suptitle(f"u_target vs u_pred", fontsize=16)

        # draw u_target for each subplot
        for i in range(6):
            axes[i].plot(u_target[:, i], label='u_target', color='black', linestyle='--', linewidth=2)
            axes[i].set_ylabel(f"u_{i}")
            axes[i].grid(True, alpha=0.3)

        # u_preq for each subplot
        for keras_model in self.list_models_keras:
            print(f"Inference calculation for {keras_model.model_name}...")
            
            u_pred = keras_model.get_predictions()
            
            # draw pred for each subplot
            for i in range(6):
                axes[i].plot(u_pred[:, i], label=f'{keras_model.model_name}', alpha=0.7)

            axes[-1].set_xlabel("Sample")
        axes[0].legend(loc='upper right') # <- to add at the botton
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            full_save_path = f"{save_path}/signals_comparison.png"
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            plt.savefig(full_save_path)
            print(f"Saved in: {full_save_path}")

        plt.show()

        
