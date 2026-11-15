import os
import matplotlib.pyplot as plt
import numpy as np

# ----- plot functions -----
def plot_cartesian(title: str, x, y, labels: list = None, colors: list = None,
                   x_label: str = "", y_label: str = "", log_y: bool = False,
                   save_path: str = None, show: bool = True) -> None:
    # one curve or a list of curves
    curves = y if isinstance(y, list) else [y]
    if labels is None:
        labels = [None] * len(curves)
    if colors is None:
        colors = [None] * len(curves) # None = matplotlib chooses

    fig, ax = plt.subplots(figsize=(9, 5))
    for curve, label, color in zip(curves, labels, colors):
        ax.plot(x, curve, label=label, color=color)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if any(label is not None for label in labels):
        n_labels = sum(label is not None for label in labels)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  ncol=n_labels, frameon=False)
    fig.tight_layout()

    if save_path is not None:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

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

    values_at_marker = []
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

    if save_path is not None:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

def plot_parity(title: str, x: list, y: list, labels: list = None, subplot_titles: list = None,
                texts: list = None, x_label: str = "", y_label: str = "", limits: tuple = (0.0, 1.0),
                point_size: float = 2, alpha: float = 0.3, ncols: int = 3,
                save_path: str = None, show: bool = True) -> None:
    """Parity plot: one subplot per panel k, one group of points (series) per s, e.g. one per model.
    On the bisector y = x the prediction (y) equals the reference (x).
      x[s][k], y[s][k]: points of series s in subplot k
      labels[s]:        name of series s, in one legend under the figure
      texts[s][k]:      text of series s in subplot k (upper left corner, same color as its points)"""
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

    if save_path is not None:
        save_plot(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

# ----- save functions -----
def save_plot(fig: plt.Figure, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved in: {save_path}")

