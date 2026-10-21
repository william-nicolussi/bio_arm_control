import os
import json
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

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

    def get_predictions(self, W: float = None, angle_range: tuple = None,
                    torque_norm_range: tuple = None) -> pd.DataFrame:
        df = self.predictions
        if W is not None:
            df = df[np.isclose(df["W"], W)]
        if angle_range is not None:
            df = df[df["angle"].between(angle_range[0], angle_range[1])]
        if torque_norm_range is not None:
            df = df[df["torque_norm"].between(torque_norm_range[0], torque_norm_range[1])]
        return df.reset_index(drop=True)
    
    def n_muscles(self) -> int:
        return sum(col.startswith("u_target_") for col in self.predictions.columns)  
 

class NN_models_visualizer:
    MUSCLE_NAMES = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]

    def __init__(self, models_to_visualize: list):
        self.list_models = []
        for model in models_to_visualize:
            self.add_model(model)

    def add_model(self, path_dir: str) -> None:
        new_model = NN_model_container(path_dir)
        self.list_models.append(new_model)

    def plot_history(self, metric: str = "loss", save_path: str = None, log_scale: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
 
        for model in self.list_models:
            if metric not in model.history.columns:
                print(f"[warning] '{metric}' not in history of {model.model_name}, skipped")
                continue
            ax.plot(
                model.history["epoch"],
                model.history[metric],
                label=f"{model.model_name} ({model.hyperparameters.get('num_parameters')} param)",
            )
 
        ax.set_title(f"Training {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
 
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            full_save_path = os.path.join(save_path, f"{metric}_comparison.png")
            fig.savefig(full_save_path, dpi=150, bbox_inches="tight")
            print(f"Graph {metric} saved in: {full_save_path}")
 
        plt.show()

    def print_table(self, metrics_in_table: list = None, split: str = "train") -> None:
 
        # if method called empty, add all scalar metrics (keeping json order)
        if not metrics_in_table:
            metrics_in_table = []
            for model in self.list_models:
                metrics = model.validation_metrics if split == "validation" else model.train_metrics
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and name not in metrics_in_table:
                        metrics_in_table.append(name)
 
        # populate table
        data = []
        for model in self.list_models:
            metrics = model.validation_metrics if split == "validation" else model.train_metrics
            row_data = {"Model Name": model.model_name}
            for metric in metrics_in_table:
                row_data[metric] = metrics.get(metric, None)
            data.append(row_data)
 
        df = pd.DataFrame(data)
        formatted_df = df.style.format(precision=4, na_rep="N/A")
        display(formatted_df)

    def plot_u_pred_u_target(self, W: float = None, angle_range: tuple = None,
                         torque_norm_range: tuple = None, muscles: list = None,
                         save_path: str = None) -> None:
 
        # u_target is the same for every model: take it from the first one
        ref_df = self.list_models[0].get_predictions(W, angle_range, torque_norm_range)
        if ref_df.empty:
            print("No samples left after filtering")
            return
 
        if muscles is None:
            muscles = list(range(self.list_models[0].n_muscles()))
 
        fig, axes = plt.subplots(len(muscles), 1, figsize=(12, 3 * len(muscles)),
                                sharex=True, squeeze=False)
        axes = axes[:, 0]
 
        # u_target, dashed black
        for ax, i in zip(axes, muscles):
            ax.plot(ref_df[f"u_target_{i}"], label="u_target", color="red", linestyle="--", linewidth=2)
            ax.set_ylabel(f"u_{i}")
            ax.grid(True, alpha=0.3)
 
        # u_pred of every model
        for model in self.list_models:
            df = model.get_predictions(W, angle_range, torque_norm_range)
            for ax, i in zip(axes, muscles):
                ax.plot(df[f"u_pred_{i}"], label=model.model_name, alpha=0.7)
 
        title = "u_target vs u_pred"
        if W is not None:
            title += f"  W={W}"
        if angle_range is not None:
            title += f"  angle in {angle_range}"
        if torque_norm_range is not None:
            title += f"  torque_norm in {torque_norm_range}"
        fig.suptitle(title, fontsize=14)
 
        axes[-1].set_xlabel("Sample")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 0.0),
           ncol=len(labels),
           frameon=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
 
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            full_save_path = os.path.join(save_path, "u_pred_u_target.png")
            fig.savefig(full_save_path, dpi=150, bbox_inches="tight")
            print(f"Saved in: {full_save_path}")
 
        plt.show()

    def plot_u_raw_u_pred(self, W: float = None, angle_range: tuple = None,
                          torque_norm_range: tuple = None, muscles: list = None,
                          save_path: str = None) -> None:
 
        ref_df = self.list_models[0].get_predictions(W, angle_range, torque_norm_range)
        if ref_df.empty:
            print("No samples left after filtering")
            return
 
        if muscles is None:
            muscles = list(range(self.list_models[0].n_muscles()))
 
        fig, axes = plt.subplots(len(muscles), 1, figsize=(12, 3 * len(muscles)),
                                sharex=True, squeeze=False)
        axes = axes[:, 0]
 
        for model in self.list_models:
            df = model.get_predictions(W, angle_range, torque_norm_range)
            for ax, i in zip(axes, muscles):
                ax.plot(df[f"u_raw_{i}"], label=f"{model.model_name} (raw)", linestyle="--", alpha=0.7)
                
                ax.plot(df[f"u_pred_{i}"], label=f"{model.model_name} (pred)", alpha=0.7)
                
                ax.set_ylabel(f"u_{i}")
                ax.grid(True, alpha=0.3)
 
        title = "u_raw vs u_pred"
        if W is not None:
            title += f"  W={W}"
        if angle_range is not None:
            title += f"  angle in {angle_range}"
        if torque_norm_range is not None:
            title += f"  torque_norm in {torque_norm_range}"
        fig.suptitle(title, fontsize=14)
 
        axes[-1].set_xlabel("Sample")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 0.0),
           ncol=len(labels),
           frameon=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
 
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            full_save_path = os.path.join(save_path, "u_raw_u_pred.png")
            fig.savefig(full_save_path, dpi=150, bbox_inches="tight")
            print(f"Saved in: {full_save_path}")
 
        plt.show()

    @staticmethod
    def _filters_title(W, angle_range, torque_norm_range) -> str:
        title = ""
        if W is not None:
            title += f"  W={W}"
        if angle_range is not None:
            title += f"  angle in {angle_range}"
        if torque_norm_range is not None:
            title += f"  torque_norm in {torque_norm_range}"
        return title
 
    def plot_parity(self, W: float = None, angle_range: tuple = None,
                    torque_norm_range: tuple = None, muscles: list = None,
                    muscle_names: list = None, max_points: int = 20000,
                    ncols: int = 3, point_size: float = 4, alpha: float = None,
                    save_path: str = None, seed: int = 0) -> None:
        """
        Parity plot (u_pred vs u_target), one subplot per muscle.
        Every model has its own color; metrics (RMSE, R2) are computed on ALL
        filtered samples, while at most `max_points` samples per model are drawn
        (None = draw everything).
        """
        n_muscles = self.list_models[0].n_muscles()
        if muscles is None:
            muscles = list(range(n_muscles))
        if muscle_names is None and len(self.MUSCLE_NAMES) == n_muscles:
            muscle_names = self.MUSCLE_NAMES
        if alpha is None:
            alpha = 0.35 if len(self.list_models) == 1 else 0.25
 
        # load (and subsample) the data of every model once
        data = []
        for m_idx, model in enumerate(self.list_models):
            df_full = model.get_predictions(W, angle_range, torque_norm_range)
            if df_full.empty:
                print(f"[warning] no samples for {model.model_name} after filtering, skipped")
                continue
            df_plot = df_full
            if max_points is not None and len(df_full) > max_points:
                df_plot = df_full.sample(n=max_points, random_state=seed)
            data.append((model, df_full, df_plot, f"C{m_idx % 10}"))
 
        if not data:
            print("No samples left after filtering")
            return
 
        nrows = int(np.ceil(len(muscles) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.ravel()
 
        for k, i in enumerate(muscles):
            ax = axes_flat[k]
            lo, hi = np.inf, -np.inf
 
            for m_pos, (model, df_full, df_plot, color) in enumerate(data):
                t = df_full[f"u_target_{i}"].to_numpy()
                p = df_full[f"u_pred_{i}"].to_numpy()
                err = p - t
                rmse = np.sqrt(np.mean(err ** 2))
                ss_tot = np.sum((t - t.mean()) ** 2)
                r2 = 1 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan
 
                ax.scatter(df_plot[f"u_target_{i}"], df_plot[f"u_pred_{i}"],
                           s=point_size, alpha=alpha, color=color,
                           edgecolors="none", rasterized=True)
 
                # metrics box, one line per model, same color as its points
                ax.text(0.03, 0.97 - 0.075 * m_pos,
                        f"RMSE {rmse:.3f}   R\u00b2 {r2:.4f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
 
                lo = min(lo, t.min(), p.min())
                hi = max(hi, t.max(), p.max())
 
            pad = 0.03 * (hi - lo) if hi > lo else 0.05
            lims = (lo - pad, hi + pad)
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
 
            name = f"u({i}): {muscle_names[i]}" if muscle_names else f"u({i})"
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("target")
            ax.set_ylabel("prediction")
 
        # hide unused subplots
        for ax in axes_flat[len(muscles):]:
            ax.set_visible(False)
 
        title = "Parity: u_pred vs u_target"
        if len(data) == 1:
            title = f"{data[0][0].model_name}: " + title
        title += self._filters_title(W, angle_range, torque_norm_range) or "  (all W)"
        fig.suptitle(title, fontsize=14, fontweight="bold")
 
        # legend with opaque, readable markers
        handles = [Line2D([], [], marker="o", linestyle="", color=color, markersize=8,
                          label=f"{model.model_name} ({model.hyperparameters.get('num_parameters')} param)")
                   for model, _, _, color in data]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 3), frameon=False)
        legend_rows = int(np.ceil(len(handles) / 3))
        plt.tight_layout(rect=[0, 0.03 * legend_rows, 1, 0.97])
 
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            suffix = f"_W_{W}".replace(".", "_") if W is not None else "_all_W"
            full_save_path = os.path.join(save_path, f"parity{suffix}.png")
            fig.savefig(full_save_path, dpi=150, bbox_inches="tight")
            print(f"Saved in: {full_save_path}")
 
        plt.show()

