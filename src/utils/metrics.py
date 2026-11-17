import numpy as np
import pandas as pd


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


def torque_metrics(lut, u: np.ndarray) -> dict:
    """Torque error |tau_0 + K·u - tau_ref| [Nm] of u (n_W, A, V, T, M) at every grid point of lut.
    Points where u is not finite (e.g. not interpolable) are excluded."""
    torque = lut.tau_0[None, :, None, None] + np.einsum("avm,wavtm->wavt", lut.k, u)
    error = np.abs(torque - lut.tau[None, :, :, :])
    error = error[np.isfinite(error)]
    return {"torque_err_mean_Nm": float(np.mean(error)),
            "torque_err_p95_Nm": float(np.percentile(error, 95)),
            "torque_err_max_Nm": float(np.max(error))}


def evaluate_grid(u_pred: np.ndarray, lut) -> dict:
    """Metrics of any prediction u_pred (n_W, A, V, T, M) against the exact u of lut, at its grid points:
    errors on u and torque error. Rows with a non-finite u_pred are excluded and counted."""
    n_outputs = u_pred.shape[-1]
    u_flat = u_pred.reshape(-1, n_outputs)
    finite = np.all(np.isfinite(u_flat), axis=1)
    metrics = compute_metrics(lut.activations.reshape(-1, n_outputs)[finite], u_flat[finite])
    metrics.update(torque_metrics(lut, u_pred))
    metrics["n_excluded"] = int(np.sum(~finite))
    return metrics


# ----- table -----
# default columns of metrics_table: key in the metrics dict -> column title
TABLE_COLUMNS = {"rmse": "RMSE u", "mae": "MAE u", "max_error": "max |err| u", "bias": "bias u",
                 "r2": "R²", "fit_pct": "FIT [%]",
                 "torque_err_mean_Nm": "τ err mean [Nm]", "torque_err_p95_Nm": "τ err p95 [Nm]",
                 "torque_err_max_Nm": "τ err max [Nm]"}

# default formats, by column title (for DataFrame.style.format)
TABLE_FORMATS = {"RMSE u": "{:.4f}", "MAE u": "{:.4f}", "max |err| u": "{:.4f}", "bias u": "{:+.1e}",
                 "R²": "{:.5f}", "FIT [%]": "{:.2f}",
                 "τ err mean [Nm]": "{:.3f}", "τ err p95 [Nm]": "{:.3f}", "τ err max [Nm]": "{:.3f}"}


def metrics_table(rows: dict, columns: dict = None) -> pd.DataFrame:
    """One row per entry of rows ({label: metrics dict}); columns: {metric key: column title}
    (default TABLE_COLUMNS). Missing metrics are NaN."""
    columns = TABLE_COLUMNS if columns is None else columns
    table = pd.DataFrame.from_dict(
        {label: {key: metrics.get(key, np.nan) for key in columns} for label, metrics in rows.items()},
        orient="index")
    return table.rename(columns=columns)