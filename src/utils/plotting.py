import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
                ncols: int = 2, legend_max_cols: int = 4, colors: list = None, linestyles: list = None,
                linestyle_legend: list = None, panel_size: tuple = (5.0, 2.8),
                save_path: str = None, show: bool = True) -> None:
    n_series = len(y)
    n_plots = len(y[0])
    labels = labels if labels is not None else [None] * n_series
    subplot_titles = subplot_titles if subplot_titles is not None else [""] * n_plots
    colors = colors if colors is not None else [f"C{s}" for s in range(n_series)]
    linestyles = linestyles if linestyles is not None else ["-"] * n_series

    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows + 0.6),
                             sharex=True, squeeze=False, layout="constrained")
    axes = axes.ravel()

    for k in range(n_plots):
        ax = axes[k]
        for s in range(n_series):
            ax.plot(x, y[s][k], color=colors[s], linestyle=linestyles[s], label=labels[s] if k == 0 else None)
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
    for label, linestyle in (linestyle_legend or []):
        handles.append(Line2D([], [], color="k", linestyle=linestyle))
        legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="outside lower center",
                   ncol=min(len(handles), legend_max_cols), frameon=False)

    _finish(fig, save_path, show)


def plot_xy(title: str, curves: list, labels: list = None, colors: list = None, linestyles: list = None,
            linestyle_legend: list = None, points: list = None, x_label: str = "", y_label: str = "",
            equal_aspect: bool = True, legend_max_cols: int = 4,
            save_path: str = None, show: bool = True) -> None:
    """Curves in a plane, each with its own x: curves = [(x, y), ...].
    points = [(x, y, marker, color), ...]: single markers (e.g. start and targets)."""
    n_curves = len(curves)
    labels = labels if labels is not None else [None] * n_curves
    colors = colors if colors is not None else [None] * n_curves
    linestyles = linestyles if linestyles is not None else ["-"] * n_curves

    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    for (x, y), label, color, linestyle in zip(curves, labels, colors, linestyles):
        ax.plot(x, y, label=label, color=color, linestyle=linestyle)
    for x, y, marker, color in (points or []):
        ax.plot(x, y, marker=marker, color=color, markersize=10, markerfacecolor="none", linestyle="none")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    handles, legend_labels = ax.get_legend_handles_labels()
    for label, linestyle in (linestyle_legend or []):
        handles.append(Line2D([], [], color="k", linestyle=linestyle))
        legend_labels.append(label)
    if handles:
        fig.legend(handles, legend_labels, loc="outside lower center",
                   ncol=min(len(handles), legend_max_cols), frameon=False)

    _finish(fig, save_path, show)


def plot_images(title: str, images: list, subplot_titles: list = None, ncols: int = None,
                image_width_in: float = 4.5, save_path: str = None, show: bool = True) -> None:
    """Images (H, W, 3) side by side, without axes: e.g. renderings of MuJoCo from several cameras."""
    n_images = len(images)
    ncols = ncols or n_images
    nrows = int(np.ceil(n_images / ncols))
    height_in = image_width_in * images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(image_width_in * ncols, height_in * nrows + 0.5),
                             squeeze=False, layout="constrained")
    axes = axes.ravel()
    subplot_titles = subplot_titles if subplot_titles is not None else [""] * n_images
    for ax, image, subplot_title in zip(axes, images, subplot_titles):
        ax.imshow(image)
        ax.set_title(subplot_title)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=12)
    _finish(fig, save_path, show)


# ----- frames of a video -----
def panel_frames(t, panels: list, title: str = "", spans: list = None, height_px: int = 480,
                 panel_width_in: float = 5.0, dpi: int = 100, span_alpha: float = 0.12) -> list:
    """One image (H, W, 3) per time of t [ms]: the panels drawn up to that time, with a cursor and a clock
    (the plots next to the frames of a video or of a player).
    panels: [{"title", "ylabel", "x" (n,) [ms], "curves": [(y (n,), label, color, linestyle), ...]}, ...]:
    one subplot each, same time axis. spans: [(t_start, t_end, label, color), ...] as in the other plots.
    height_px: height of the images, at least what the panels need. Fast: axes, grid, phases and legends are
    drawn once (background), and at every time only the lines, the cursors and the clock (blitting)."""
    fig_h = max(height_px / dpi, 1.6 * len(panels) + 0.9)   # room for the panels
    fig = plt.figure(figsize=(panel_width_in, fig_h), dpi=dpi)
    canvas = FigureCanvasAgg(fig)

    left, right = 0.7 / panel_width_in, 1.0 - 0.15 / panel_width_in
    top, bottom, gap = 1.0 - 1.0 / fig_h, 0.55 / fig_h, 0.45 / fig_h
    height = (top - bottom - gap * (len(panels) - 1)) / len(panels)
    fig.text(0.5, 1.0 - 0.08 / fig_h, title, ha="center", va="top", fontsize=9)
    clock = fig.text(0.5, 1.0 - 0.4 / fig_h, "", ha="center", va="top", fontsize=12, fontweight="bold",
                     animated=True)

    live = []       # (line, x, y) updated at every frame
    cursors = []
    for p, panel in enumerate(panels):
        ax = fig.add_axes((left, top - (p + 1) * height - p * gap, right - left, height))
        x = np.asarray(panel["x"], dtype=float)
        values = np.concatenate([np.asarray(c[0], dtype=float) for c in panel["curves"]])
        values = values[np.isfinite(values)]
        lo, hi = values.min(), values.max()
        margin = 0.1 * (hi - lo) if hi > lo else 1.0
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(lo - margin, hi + margin)
        for x_start, x_end, _, color in spans or []:
            ax.axvspan(x_start, x_end, color=color, alpha=span_alpha, linewidth=0)
        for y, label, color, linestyle in panel["curves"]:
            line, = ax.plot([], [], color=color, linestyle=linestyle, label=label, animated=True)
            live.append((ax, line, x, np.asarray(y, dtype=float)))
        cursors.append((ax, ax.axvline(x[0], color="0.4", linewidth=0.8, animated=True)))
        ax.set_title(panel.get("title", ""), fontsize=9, pad=3)
        ax.set_ylabel(panel.get("ylabel", ""), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if p == len(panels) - 1:
            ax.set_xlabel("Time [ms]", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        if any(c[1] for c in panel["curves"]):
            # legend from proxies: the real lines are animated, so they are not in the background
            handles = [Line2D([], [], color=c[2], linestyle=c[3]) for c in panel["curves"]]
            ax.legend(handles, [c[1] for c in panel["curves"]], fontsize=7, loc="upper right")

    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)

    frames = []
    for t_k in np.asarray(t, dtype=float):
        canvas.restore_region(background)
        for ax, line, x, y in live:
            mask = x <= t_k
            line.set_data(x[mask], y[mask])
            ax.draw_artist(line)
        for ax, cursor in cursors:
            cursor.set_xdata([t_k, t_k])
            ax.draw_artist(cursor)
        clock.set_text(f"t = {t_k:.0f} ms")
        fig.draw_artist(clock)
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return frames


# ----- text -----
def values_text(values) -> str:
    """Values for a title: one value as it is (0.04), more in brackets ([4, 2])."""
    values = np.atleast_1d(values)
    return f"{values[0]:g}" if len(values) == 1 else "[" + ", ".join(f"{v:g}" for v in values) + "]"


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