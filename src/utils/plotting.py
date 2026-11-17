import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ----- plot functions -----
def plot_cartesian(title: str, x, y, labels: list = None, colors: list = None,
                   linestyles: list = None, linestyle_legend: list = None,
                   x_label: str = "", y_label: str = "", log_x: bool = False, log_y: bool = False,
                   spans: list = None, span_alpha: float = 0.12, legend_max_cols: int = 4,
                   save_path: str = None, show: bool = True) -> None:
    curves = y if isinstance(y, list) else [y]
    n_curves = len(curves)
    labels = labels if labels is not None else [None] * n_curves
    colors = colors if colors is not None else [None] * n_curves
    linestyles = linestyles if linestyles is not None else ["-"] * n_curves

    # constrained layout makes room for the legend outside the axes
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for curve, label, color, linestyle in zip(curves, labels, colors, linestyles):
        ax.plot(x, curve, label=label, color=color, linestyle=linestyle)
    _draw_spans(ax, spans or [], span_alpha, with_labels=True)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    handles, legend_labels = ax.get_legend_handles_labels()
    for label, linestyle in (linestyle_legend or []):
        handles.append(Line2D([], [], color="k", linestyle=linestyle))
        legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="outside lower center",
                   ncol=min(len(handles), legend_max_cols), frameon=False)

    _finish(fig, save_path, show)


def plot_heatmaps(title: str, x, y, z: list, subplot_titles: list = None,
                  x_label: str = "", y_label: str = "", colorbar_label: str = "",
                  cmap: str = "viridis", vmin: float = None, vmax: float = None,
                  centered: bool = False, x_req: float = None, y_req: float = None,
                  marker_color: str = "red", ncols: int = 3,
                  save_path: str = None, show: bool = True) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    n_plots = len(z)
    if subplot_titles is None:
        subplot_titles = [""] * n_plots

    # one color scale for all subplots
    all_values = np.concatenate([np.ravel(z_k) for z_k in z])
    all_values = all_values[np.isfinite(all_values)]
    if centered:
        max_abs = np.max(np.abs(all_values))
        vmin, vmax = -max_abs, max_abs
    else:
        if vmin is None:
            vmin = np.min(all_values)
        if vmax is None:
            vmax = np.max(all_values)

    # marker on the nearest grid node, only if the point is inside the grid
    show_marker = x_req is not None and y_req is not None
    if show_marker:
        if x.min() <= x_req <= x.max() and y.min() <= y_req <= y.max():
            ix = np.argmin(np.abs(x - x_req))
            iy = np.argmin(np.abs(y - y_req))
        else:
            print(f"Point ({x_req:g}, {y_req:g}) outside the grid: no marker")
            show_marker = False

    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, layout="constrained")
    axes = axes.ravel()

    for k in range(n_plots):
        ax = axes[k]
        z_k = np.asarray(z[k])
        # z_k is (len(x), len(y)), pcolormesh wants (len(y), len(x)): transpose
        mesh = ax.pcolormesh(x, y, z_k.T, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading="nearest")
        subplot_title = subplot_titles[k]
        if show_marker:
            value = z_k[ix, iy]
            ax.plot(x[ix], y[iy], marker="s", color=marker_color,
                    markersize=10, alpha=0.6)
            subplot_title += f": {value:.3f}"
        ax.set_title(subplot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    for ax in axes[n_plots:]:
        ax.set_visible(False)   # empty cells of the last row

    fig.suptitle(title, fontsize=14)
    cbar = fig.colorbar(mesh, ax=axes[:n_plots].tolist(), shrink=0.8)
    cbar.set_label(colorbar_label)

    _finish(fig, save_path, show)


def plot_parity(title: str, x: list, y: list, labels: list = None, subplot_titles: list = None,
                texts: list = None, x_label: str = "", y_label: str = "", limits: tuple = (0.0, 1.0),
                point_size: float = 2, alpha: float = 0.3, ncols: int = 3,
                save_path: str = None, show: bool = True) -> None:
    n_series = len(x)
    n_plots = len(x[0])
    if labels is None:
        labels = [None] * n_series
    if subplot_titles is None:
        subplot_titles = [""] * n_plots
    if texts is None:
        texts = [[""] * n_plots for _ in range(n_series)]

    # small margin, so the points on the limits (e.g. u = 0 and u = 1) are not cut
    margin = 0.02 * (limits[1] - limits[0])
    axis_limits = (limits[0] - margin, limits[1] + margin)

    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False, layout="constrained")
    axes = axes.ravel()

    handles = []    # one scatter per series, for the legend
    for k in range(n_plots):
        ax = axes[k]
        for s in range(n_series):
            color = f"C{s}"     # same color for series s in every subplot
            points = ax.scatter(x[s][k], y[s][k], s=point_size, alpha=alpha, linewidths=0, color=color)
            if k == 0:
                handles.append(points)
            if texts[s][k]:
                # one line per series, one under the other
                ax.text(0.03, 0.97 - 0.07 * s, texts[s][k], transform=ax.transAxes,
                        va="top", fontsize=9, fontweight="bold", color=color)
        ax.plot(limits, limits, "k--", linewidth=1)          # bisector: perfect prediction
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_aspect("equal")
        ax.set_title(subplot_titles[k])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_plots:]:
        ax.set_visible(False)   # empty cells of the last row

    fig.suptitle(title, fontsize=14)
    if any(label is not None for label in labels):
        # one legend for the whole figure, under the subplots; bigger and opaque markers, readable
        shown = [(h, label) for h, label in zip(handles, labels) if label is not None]
        legend = fig.legend([h for h, _ in shown], [label for _, label in shown],
                            loc="outside lower center", ncol=len(shown),
                            markerscale=6, frameon=False)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)

    _finish(fig, save_path, show)


def plot_stacked(title: str, x, y: list, y_labels: list = None, x_label: str = "",
                 spans: list = None, color: str = "k", span_alpha: float = 0.12,
                 save_path: str = None, show: bool = True) -> None:
    n_plots = len(y)
    if y_labels is None:
        y_labels = [""] * n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 2.2 * n_plots + 0.6), sharex=True,
                             squeeze=False, layout="constrained")
    axes = axes.ravel()

    for k, (ax, curve, y_label) in enumerate(zip(axes, y, y_labels)):
        ax.plot(x, curve, color=color)
        _draw_spans(ax, spans or [], span_alpha, with_labels=(k == 0))
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(x_label)

    fig.suptitle(title, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)

    _finish(fig, save_path, show)


def plot_panels(title: str, x, y: list, labels: list = None, subplot_titles: list = None,
                x_label: str = "", y_label: str = "", spans: list = None, span_alpha: float = 0.12,
                ncols: int = 2, legend_max_cols: int = 4,
                save_path: str = None, show: bool = True) -> None:
    n_series = len(y)
    n_plots = len(y[0])
    labels = labels if labels is not None else [None] * n_series
    subplot_titles = subplot_titles if subplot_titles is not None else [""] * n_plots

    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows + 0.6), sharex=True,
                             squeeze=False, layout="constrained")
    axes = axes.ravel()

    for k in range(n_plots):
        ax = axes[k]
        for s in range(n_series):
            ax.plot(x, y[s][k], color=f"C{s}", label=labels[s] if k == 0 else None)
        _draw_spans(ax, spans or [], span_alpha, with_labels=(k == 0))
        ax.set_title(subplot_titles[k])
        ax.grid(True, alpha=0.3)
        if k % ncols == 0:
            ax.set_ylabel(y_label)
        if k + ncols >= n_plots:
            ax.set_xlabel(x_label)     # last subplot of its column

    for ax in axes[n_plots:]:
        ax.set_visible(False)   # empty cells of the last row

    fig.suptitle(title, fontsize=14)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="outside lower center",
                   ncol=min(len(handles), legend_max_cols), frameon=False)

    _finish(fig, save_path, show)


# ----- save functions -----
def save_plot(fig: plt.Figure, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved in: {save_path}")


# ----- helpers -----
def _draw_spans(ax, spans: list, alpha: float, with_labels: bool) -> None:
    for x_start, x_end, label, color in spans:
        ax.axvspan(x_start, x_end, color=color, alpha=alpha, linewidth=0,
                   label=label if with_labels else None)


def _finish(fig: plt.Figure, save_path: str, show: bool) -> None:
    if save_path is not None:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)