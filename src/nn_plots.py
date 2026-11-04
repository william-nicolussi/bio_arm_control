"""Plots of the NN that approximates the LUT.

Every function works only on predictions DataFrames with the columns
    W, angle, torque_norm, u_target_i, u_pred_i  (and u_raw_i, optional),  i = 0...N-1
and never computes predictions itself. Pass either one DataFrame or a dict
{label: DataFrame}: the label is used in legends and titles (e.g. "model_A (1234 param)").

The callers (NN_model_container, NN_validator, NN_model_from_LUT) build the DataFrames
and choose file_name, so the names of the saved files stay under their control.
Filters (W, angle_range, torque_norm_range) are applied by filter_predictions().
"""
import os
from typing import Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.nn_metrics import columns, compute_metrics, muscle_label, n_muscles, rmse_r2

Frames = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]

TARGET_STYLE = dict(color="black", linestyle="--", linewidth=1, zorder=3)
DEFAULT_LABEL = "NN"


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def filter_predictions(df: pd.DataFrame, W=None, angle_range: tuple = None,
                       torque_norm_range: tuple = None) -> pd.DataFrame:
    """Rows of df with W (a value or a list of values), angle and torque_norm in the given
    closed ranges. None = no filter. The index is reset (x axis = sample)."""
    mask = np.ones(len(df), dtype=bool)
    if W is not None:
        W_values = np.atleast_1d(np.asarray(W, dtype=float))
        mask &= np.isclose(df["W"].to_numpy(dtype=float)[:, None], W_values[None, :]).any(axis=1)
    if angle_range is not None:
        mask &= df["angle"].between(angle_range[0], angle_range[1]).to_numpy()
    if torque_norm_range is not None:
        mask &= df["torque_norm"].between(torque_norm_range[0], torque_norm_range[1]).to_numpy()
    return df[mask].reset_index(drop=True)


def filters_title(W=None, angle_range: tuple = None, torque_norm_range: tuple = None) -> str:
    """Text describing the filters, e.g. '  W=0.5  angle in (30, 90)'. Empty if no filter."""
    title = ""
    if W is not None:
        W_text = ", ".join(f"{w:g}" for w in np.atleast_1d(W))
        title += f"  W={W_text}"
    if angle_range is not None:
        title += f"  angle in {tuple(angle_range)}"
    if torque_norm_range is not None:
        title += f"  torque_norm in {tuple(torque_norm_range)}"
    return title


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _as_dict(dfs: Frames) -> dict:
    if isinstance(dfs, pd.DataFrame):
        return {DEFAULT_LABEL: dfs}
    frames = dict(dfs)
    if not frames:
        raise ValueError("No DataFrame given")
    return frames


def _colors(frames: dict) -> dict:
    """One color per label, fixed by the order of the input (stable even if a frame is skipped)."""
    return {label: f"C{k % 10}" for k, label in enumerate(frames)}


def _prepare(dfs: Frames, W, angle_range, torque_norm_range):
    """Filter every frame; frames left empty are skipped with a warning."""
    frames = _as_dict(dfs)
    colors = _colors(frames)
    out = {}
    for label, df in frames.items():
        df_f = filter_predictions(df, W, angle_range, torque_norm_range)
        if df_f.empty:
            print(f"[warning] no samples for '{label}' after filtering, skipped")
            continue
        out[label] = df_f
    if not out:
        print("No samples left after filtering")
    return out, colors


def _muscles(df: pd.DataFrame, muscles) -> tuple:
    n = n_muscles(df)
    if muscles is None:
        muscles = list(range(n))
    return list(muscles), n


def _make_title(base: str, frames: dict, W, angle_range, torque_norm_range, title: str = None) -> str:
    if title is not None:
        return title
    if len(frames) == 1:
        label = next(iter(frames))
        if label != DEFAULT_LABEL:
            base = f"{label}: {base}"
    return base + (filters_title(W, angle_range, torque_norm_range) or "  (all W)")


def _check_same_target(frames: dict, muscles: list) -> None:
    """The target is drawn once (from the first frame): warn if the others differ."""
    ref_label, ref = next(iter(frames.items()))
    cols = columns("u_target", muscles)
    for label, df in list(frames.items())[1:]:
        if len(df) != len(ref) or not np.allclose(df[cols].to_numpy(), ref[cols].to_numpy(),
                                                  equal_nan=True):
            print(f"[warning] u_target of '{label}' differs from '{ref_label}': "
                  f"only the target of '{ref_label}' is drawn")


def _legend_below(fig, handles, labels, max_cols: int = 4) -> float:
    """Legend under the figure; returns the bottom margin to leave in tight_layout."""
    if not handles:
        return 0.0
    ncol = min(len(handles), max_cols)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False)
    rows = int(np.ceil(len(handles) / ncol))
    return 0.03 * rows


def _finish(fig, save_path: str, file_name: str, show: bool) -> None:
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if not file_name.lower().endswith(".png"):
            file_name += ".png"
        full_path = os.path.join(save_path, file_name)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved in: {full_path}")
    if show:
        plt.show()
    plt.close(fig)  # free memory (important in a grid search with many plots)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_series(dfs: Frames, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                muscles: list = None, error: bool = False, title: str = None,
                save_path: str = None, file_name: str = None, show: bool = True) -> None:
    """One subplot per muscle along the samples.
    error=False: u_target (red, dashed, from the first frame) and u_pred of every frame.
    error=True:  u_pred - u_target of every frame."""
    frames, colors = _prepare(dfs, W, angle_range, torque_norm_range)
    if not frames:
        return
    muscles, n = _muscles(next(iter(frames.values())), muscles)

    fig, axes = plt.subplots(len(muscles), 1, figsize=(12, 2.6 * len(muscles)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    if not error:
        _check_same_target(frames, muscles)
        ref = next(iter(frames.values()))
        for ax, i in zip(axes, muscles):
            ax.plot(ref[f"u_target_{i}"].to_numpy(), label="u_target", **TARGET_STYLE)

    for label, df in frames.items():
        for ax, i in zip(axes, muscles):
            y = df[f"u_pred_{i}"].to_numpy()
            if error:
                y = y - df[f"u_target_{i}"].to_numpy()
            ax.plot(y, label=label, color=colors[label], linewidth=1.0, alpha=0.8)

    for ax, i in zip(axes, muscles):
        if error:
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(muscle_label(i, n))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sample")

    base = "error = u_pred - u_target" if error else "u_target vs u_pred"
    fig.suptitle(_make_title(base, frames, W, angle_range, torque_norm_range, title), fontsize=14)
    bottom = _legend_below(fig, *axes[0].get_legend_handles_labels())
    fig.tight_layout(rect=(0, bottom, 1, 0.97))

    if file_name is None:
        file_name = "u_pred_error" if error else "u_pred_u_target"
    _finish(fig, save_path, file_name, show)


def plot_parity(dfs: Frames, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                muscles: list = None, max_points: int = 20000, ncols: int = 3,
                point_size: float = 4, alpha: float = None, seed: int = 0, title: str = None,
                save_path: str = None, file_name: str = None, show: bool = True) -> None:
    """Parity plot (u_pred vs u_target), one subplot per muscle, one color per frame.
    RMSE and R2 are computed on ALL filtered samples; at most max_points samples per frame
    are drawn (None = draw everything)."""
    frames, colors = _prepare(dfs, W, angle_range, torque_norm_range)
    if not frames:
        return
    muscles, n = _muscles(next(iter(frames.values())), muscles)
    if alpha is None:
        alpha = 0.35 if len(frames) == 1 else 0.25

    drawn = {}
    for label, df in frames.items():
        if max_points is not None and len(df) > max_points:
            drawn[label] = df.sample(n=max_points, random_state=seed)
        else:
            drawn[label] = df

    nrows = int(np.ceil(len(muscles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for k, i in enumerate(muscles):
        ax = axes_flat[k]
        lo, hi = np.inf, -np.inf
        for m_pos, (label, df) in enumerate(frames.items()):
            t = df[f"u_target_{i}"].to_numpy(dtype=float)
            p = df[f"u_pred_{i}"].to_numpy(dtype=float)
            rmse, r2 = rmse_r2(t, p)

            ax.scatter(drawn[label][f"u_target_{i}"], drawn[label][f"u_pred_{i}"],
                       s=point_size, alpha=alpha, color=colors[label],
                       edgecolors="none", rasterized=True)

            # metrics box, one line per frame, same color as its points
            ax.text(0.03, 0.97 - 0.075 * m_pos, f"RMSE {rmse:.3f}   R\u00b2 {r2:.4f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color=colors[label], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

            finite = np.isfinite(t) & np.isfinite(p)
            if finite.any():
                lo = min(lo, t[finite].min(), p[finite].min())
                hi = max(hi, t[finite].max(), p[finite].max())

        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        pad = 0.03 * (hi - lo) if hi > lo else 0.05
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(muscle_label(i, n), fontweight="bold")
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")

    for ax in axes_flat[len(muscles):]:
        ax.set_visible(False)

    fig.suptitle(_make_title("Parity: u_pred vs u_target", frames, W, angle_range,
                             torque_norm_range, title), fontsize=14, fontweight="bold")

    # legend with opaque, readable markers
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[label], markersize=8)
               for label in frames]
    bottom = _legend_below(fig, handles, list(frames), max_cols=3)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))

    if file_name is None:
        file_name = "parity"
    _finish(fig, save_path, file_name, show)


def plot_error_hist(dfs: Frames, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                    muscles: list = None, bins: int = 60, log: bool = False, title: str = None,
                    save_path: str = None, file_name: str = None, show: bool = True) -> None:
    """Histogram of u_pred - u_target, all selected muscles pooled together.
    One frame: filled histogram, bias line and global metrics under the plot.
    More frames: one outline per frame, same bins, RMSE and bias in the legend."""
    frames, colors = _prepare(dfs, W, angle_range, torque_norm_range)
    if not frames:
        return
    muscles, _ = _muscles(next(iter(frames.values())), muscles)

    errors, metrics = {}, {}
    for label, df in frames.items():
        y = df[columns("u_target", muscles)].to_numpy(dtype=float)
        p = df[columns("u_pred", muscles)].to_numpy(dtype=float)
        err = (p - y).ravel()
        errors[label] = err[np.isfinite(err)]
        metrics[label] = compute_metrics(y, p)

    edges = np.histogram_bin_edges(np.concatenate(list(errors.values())), bins=bins)
    fig, ax = plt.subplots(figsize=(9, 5))

    if len(frames) == 1:
        label = next(iter(frames))
        m = metrics[label]
        ax.hist(errors[label], bins=edges, alpha=0.7, color=colors[label], log=log)
        ax.axvline(m["bias"], color="red", label=f"bias {m['bias']:+.3e}")
        fig.text(0.5, 0.005,
                 f"RMSE {m['rmse']:.3e}   MAE {m['mae']:.3e}   max {m['max_error']:.3e}   "
                 f"FIT {m['fit_pct']:.2f}%   R2 {m['r2']:.4f}",
                 ha="center", fontsize=8, color="0.3")
    else:
        for label, err in errors.items():
            m = metrics[label]
            ax.hist(err, bins=edges, histtype="step", linewidth=1.5, color=colors[label], log=log,
                    label=f"{label}   RMSE {m['rmse']:.3e}   bias {m['bias']:+.2e}")
            ax.axvline(m["bias"], color=colors[label], linestyle="--", linewidth=1)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("error = u_pred - u_target")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(_make_title("error distribution", frames, W, angle_range, torque_norm_range, title))
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    if file_name is None:
        file_name = "error_hist"
    _finish(fig, save_path, file_name, show)


def plot_raw_vs_pred(dfs: Frames, W=None, angle_range: tuple = None, torque_norm_range: tuple = None,
                     muscles: list = None, title: str = None,
                     save_path: str = None, file_name: str = None, show: bool = True) -> None:
    """u_raw (dashed) and u_pred = clip(u_raw, 0, 1) (solid), one subplot per muscle.
    Frames without the u_raw_i columns (e.g. NN_validator) are skipped."""
    frames, colors = _prepare(dfs, W, angle_range, torque_norm_range)
    if not frames:
        return
    with_raw = {}
    for label, df in frames.items():
        if "u_raw_0" in df.columns:
            with_raw[label] = df
        else:
            print(f"[warning] '{label}' has no u_raw columns, skipped")
    if not with_raw:
        print("No DataFrame with u_raw columns")
        return
    muscles, n = _muscles(next(iter(with_raw.values())), muscles)

    fig, axes = plt.subplots(len(muscles), 1, figsize=(12, 2.6 * len(muscles)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for label, df in with_raw.items():
        for ax, i in zip(axes, muscles):
            ax.plot(df[f"u_raw_{i}"].to_numpy(), color=colors[label], linestyle="--", alpha=0.7,
                    label=f"{label} (raw)")
            ax.plot(df[f"u_pred_{i}"].to_numpy(), color=colors[label], alpha=0.7,
                    label=f"{label} (pred)")

    for ax, i in zip(axes, muscles):
        for bound in (0.0, 1.0):  # clip limits
            ax.axhline(bound, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_ylabel(muscle_label(i, n))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sample")

    fig.suptitle(_make_title("u_raw vs u_pred", with_raw, W, angle_range, torque_norm_range, title),
                 fontsize=14)
    bottom = _legend_below(fig, *axes[0].get_legend_handles_labels())
    fig.tight_layout(rect=(0, bottom, 1, 0.97))

    if file_name is None:
        file_name = "u_raw_u_pred"
    _finish(fig, save_path, file_name, show)


def plot_history(histories, metric: str = "loss", log_scale: bool = True, title: str = None,
                 save_path: str = None, file_name: str = None, show: bool = True) -> None:
    """Training curves. histories: one history or a dict {label: history}; a history is a
    DataFrame (or a dict of lists, as keras History.history) with the column `metric` and,
    optionally, `epoch` (otherwise epochs are 1...N)."""
    if isinstance(histories, pd.DataFrame):
        histories = {DEFAULT_LABEL: histories}
    histories = {label: pd.DataFrame(h) for label, h in dict(histories).items()}
    colors = _colors(histories)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = {}
    for label, h in histories.items():
        if metric not in h.columns:
            print(f"[warning] '{metric}' not in history of '{label}', skipped")
            continue
        epochs = h["epoch"].to_numpy() if "epoch" in h.columns else np.arange(1, len(h) + 1)
        ax.plot(epochs, h[metric].to_numpy(), color=colors[label], label=label)
        plotted[label] = h
    if not plotted:
        plt.close(fig)
        print(f"No history with '{metric}'")
        return

    if title is None:
        title = f"Training {metric}"
        if len(plotted) == 1 and DEFAULT_LABEL not in plotted:
            title = f"{next(iter(plotted))}: {title}"
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()

    if file_name is None:
        file_name = f"{metric}_comparison"
    _finish(fig, save_path, file_name, show)


HEATMAP_VALUES = {
    "u_pred": "u_pred",
    "u_target": "u_target",
    "u_raw": "u_raw",
    "error": "error = u_pred - u_target",
}


def plot_heatmaps(df: Frames, value: str = "error", W: float = None,
                  angle_range: tuple = None, torque_norm_range: tuple = None,
                  angle_req: float = None, tau_req: float = None, ncols: int = 3,
                  title: str = None, save_path: str = None, file_name: str = None,
                  show: bool = True) -> dict:
    """Heatmaps of `value` (u_pred, u_target, u_raw or error) over (angle, torque_norm),
    one subplot per muscle, same color scale for all. The data must contain ONE W:
    pass W= when the frame contains more.
    error: colormap bwr centered on 0 (red: prediction too high, blue: too low).
    (angle_req, tau_req): marker on the nearest grid point; tau_req is torque_norm, the
    conversion from Nm is up to the caller (it needs the LUT). Returns the values at the
    marker {muscle_label: value}, or None."""
    if value not in HEATMAP_VALUES:
        raise ValueError(f"value must be one of {list(HEATMAP_VALUES)}, got '{value}'")

    frames, _ = _prepare(df, W, angle_range, torque_norm_range)
    if not frames:
        return None
    if len(frames) > 1:
        raise ValueError("plot_heatmaps draws one DataFrame: call it once per model")
    label, data = next(iter(frames.items()))

    W_values = np.unique(data["W"])
    if len(W_values) != 1:
        raise ValueError(f"Heatmaps need one W: pass W= (available: {W_values})")
    if value == "u_raw" and "u_raw_0" not in data.columns:
        raise ValueError(f"'{label}' has no u_raw columns")
    muscles, n = _muscles(data, None)

    # one value column per muscle
    values = {}
    for i in muscles:
        if value == "error":
            values[i] = data[f"u_pred_{i}"].to_numpy() - data[f"u_target_{i}"].to_numpy()
        else:
            values[i] = data[f"{value}_{i}"].to_numpy()
    all_values = np.concatenate(list(values.values()))
    all_values = all_values[np.isfinite(all_values)]

    if value == "error":
        max_abs = np.max(np.abs(all_values)) if all_values.size else 1.0
        max_abs = max_abs if max_abs > 0 else 1.0
        vmin, vmax, cmap, marker_color = -max_abs, max_abs, "bwr", "black"
    else:
        vmin, vmax = np.min(all_values), np.max(all_values)
        cmap, marker_color = "viridis", "red"

    # torque_norm rounded: float noise between angles would split one grid row into many
    grid = pd.DataFrame({"angle": data["angle"].to_numpy(),
                         "torque_norm": np.round(data["torque_norm"].to_numpy(), 9)})

    show_marker = (angle_req is not None and tau_req is not None
                   and grid["angle"].min() <= angle_req <= grid["angle"].max()
                   and 0.0 <= tau_req <= 1.0)
    if angle_req is not None and tau_req is not None and not show_marker:
        print(f"Point (angle={angle_req}, torque_norm={tau_req}) outside the data: no marker")

    nrows = int(np.ceil(len(muscles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), layout="constrained",
                             squeeze=False)
    axes_flat = axes.ravel()

    found = {}
    im = None
    for k, i in enumerate(muscles):
        ax = axes_flat[k]
        heat = grid.assign(v=values[i]).pivot(index="torque_norm", columns="angle", values="v")
        X, Y, Z = heat.columns.values, heat.index.values, heat.values
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

        if show_marker:
            ia, it = np.abs(X - angle_req).argmin(), np.abs(Y - tau_req).argmin()
            closest_angle, closest_tau = X[ia], Y[it]
            found[muscle_label(i, n)] = Z[it, ia]
            ax.plot(closest_angle, closest_tau, marker="s", color=marker_color,
                    alpha=0.5, markersize=10)

        prefix = "error " if value == "error" else f"{value} "
        ax.set_title(prefix + muscle_label(i, n), fontweight="bold")
        ax.set_xlabel("Angle")
        ax.set_ylabel("Torque Norm")

    for ax in axes_flat[len(muscles):]:
        ax.set_visible(False)

    if title is None:
        title = f"{HEATMAP_VALUES[value]}, W = {W_values[0]:g}"
        if label != DEFAULT_LABEL:
            title = f"{label}: {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes_flat[:len(muscles)].tolist(), orientation="vertical",
                        shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    if file_name is None:
        file_name = f"{value}_heatmaps_W_{W_values[0]:g}"
    _finish(fig, save_path, file_name, show)

    if show_marker:
        print(f"Values at (angle~{closest_angle:.2f}, torque_norm~{closest_tau:.2f}):")
        for name, v in found.items():
            print(f"  {name:<16} {v:.4f}")
        return found
    return None