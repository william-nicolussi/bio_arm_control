"""Constants and error metrics shared by training, validation and plots.

No Keras import here: nn_plots, nn_validator and nn_model_from_lut can all import
this module without circular imports.
"""
import numpy as np
import pandas as pd

N_OUTPUTS = 6
MUSCLE_NAMES = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]


def columns(prefix: str, muscles=None) -> list:
    """Column names of one quantity, e.g. columns("u_pred") -> ["u_pred_0", ..., "u_pred_5"]."""
    if muscles is None:
        muscles = range(N_OUTPUTS)
    return [f"{prefix}_{i}" for i in muscles]


def n_muscles(df: pd.DataFrame) -> int:
    """Number of outputs in a predictions DataFrame (counts the u_target_i columns)."""
    return sum(col.startswith("u_target_") for col in df.columns)


def muscle_label(i: int, n: int = N_OUTPUTS) -> str:
    """'u(i): NAME' when the names are known for n outputs, else 'u(i)'."""
    if n == len(MUSCLE_NAMES):
        return f"u({i}): {MUSCLE_NAMES[i]}"
    return f"u({i})"


def compute_metrics(u_target: np.ndarray, u_pred: np.ndarray) -> dict:
    """Error metrics between target and prediction, both (N, M). Same definitions as nnodely."""
    finite = np.isfinite(u_pred).all(axis=1)
    y, p = u_target[finite], u_pred[finite]
    err = p - y

    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(err ** 2) / ss_tot

    # per muscle (column by column); a constant target column gives nan/inf, without warnings
    mse_m = np.mean(err ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_m = 1.0 - np.sum(err ** 2, axis=0) / np.sum((y - y.mean(axis=0)) ** 2, axis=0)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(np.mean(np.abs(err))),
        "max_error": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "nrmse_pct": float(100.0 * rmse / (y.max() - y.min())),
        "fit_pct": float(100.0 * (1.0 - np.sqrt(1.0 - r2))),
        "r2": float(r2),
        "correlation": float(np.corrcoef(y.ravel(), p.ravel())[0, 1]),
        "non_finite": int(np.sum(~finite)),
        "per_muscle": {
            "rmse": np.sqrt(mse_m).tolist(),
            "max_error": np.max(np.abs(err), axis=0).tolist(),
            "r2": r2_m.tolist(),
        },
    }


def metrics_from_df(df: pd.DataFrame, muscles=None) -> dict:
    """compute_metrics on the u_target_i / u_pred_i columns of a predictions DataFrame."""
    if muscles is None:
        muscles = range(n_muscles(df))
    y = df[columns("u_target", muscles)].to_numpy(dtype=float)
    p = df[columns("u_pred", muscles)].to_numpy(dtype=float)
    return compute_metrics(y, p)


def rmse_r2(t, p) -> tuple:
    """RMSE and R2 of a single output (1-D arrays). R2 is nan when the target is constant."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    finite = np.isfinite(t) & np.isfinite(p)
    t, p = t[finite], p[finite]
    if t.size == 0:
        return np.nan, np.nan
    err = p - t
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = np.sum((t - t.mean()) ** 2)
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r2