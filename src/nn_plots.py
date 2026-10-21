import os

import matplotlib.pyplot as plt
import numpy as np

from src.nn_model_from_lut import compute_metrics, N_OUTPUTS


def plot_predictions(y: np.ndarray, p: np.ndarray, title: str, file_prefix: str,
                     save_path: str = None, show: bool = False) -> None:
    """u_pred vs u_target, one figure per plot: time series, error, parity, error histogram."""
    err = p - y
    m = compute_metrics(y, p)
    sample = np.arange(len(y))
    colors = plt.cm.cool(np.linspace(0.2, 0.8, N_OUTPUTS))
    metrics_text = (f"RMSE {m['rmse']:.3e}   MAE {m['mae']:.3e}   max {m['max_error']:.3e}   "
                    f"FIT {m['fit_pct']:.2f}%   R2 {m['r2']:.4f}")
    figs = {}

    # 1. prediction (solid) vs target (dashed)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(N_OUTPUTS):
        ax.plot(sample, y[:, i], "--", color="0.25", linewidth=1.2)
        ax.plot(sample, p[:, i], color=colors[i], linewidth=1.0, label=f"u{i}")
    ax.set_xlabel("sample")
    ax.set_ylabel("value")
    ax.legend(ncol=N_OUTPUTS, fontsize=8, loc="upper right")
    ax.set_title(f"{title}: u_pred (solid) vs u_target (dashed)")
    figs["series"] = fig

    # 2. error along the samples
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i in range(N_OUTPUTS):
        ax.plot(sample, err[:, i], color=colors[i], linewidth=0.8, label=f"u{i}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("sample")
    ax.set_ylabel("prediction - target")
    ax.legend(ncol=N_OUTPUTS, fontsize=8, loc="upper right")
    ax.set_title(f"{title}: error")
    figs["error"] = fig

    # 3. parity plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(N_OUTPUTS):
        ax.scatter(y[:, i], p[:, i], s=6, alpha=0.4, color=colors[i], label=f"u{i}")
    ax.plot([0, 1], [0, 1], "--", color="0.2", linewidth=1)
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{title}: parity")
    fig.text(0.5, 0.005, metrics_text, ha="center", fontsize=8, color="0.3")
    figs["parity"] = fig

    # 4. error histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(err.ravel(), bins=60, alpha=0.7)
    ax.axvline(m["bias"], color="red", label=f"bias {m['bias']:+.3e}")
    ax.set_xlabel("error")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title(f"{title}: error distribution")
    fig.text(0.5, 0.005, metrics_text, ha="center", fontsize=8, color="0.3")
    figs["hist"] = fig

    for name, fig in figs.items():
        for ax in fig.axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, f"{file_prefix}_{name}.png")
            fig.savefig(file_path, dpi=150, bbox_inches="tight")
            print(f"Predictions plot saved in: {file_path}")
    if show:
        plt.show()
    for fig in figs.values():
        plt.close(fig)