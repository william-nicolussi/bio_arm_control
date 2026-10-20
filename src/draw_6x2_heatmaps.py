import os
import numpy as np
import matplotlib.pyplot as plt

def draw_heatmaps(
    df, 
    angle_min: float, 
    angle_max: float, 
    angle_req: float = None, 
    tau_req: float = None,
    cmap: str = 'viridis',
    title_prefix: str = 'u',
    marker_color: str = 'black',
    center_zero: bool = False,
    save_path: str = None
) -> None:
    muscle_names = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]
    
    muscle_cols = [0, 1, 2, 3, 4, 5]
    all_values = df[muscle_cols].values

    if center_zero:
        max_abs = np.max(np.abs(all_values))
        if max_abs == 0:
            max_abs = 1.0
        global_vmin, global_vmax = -max_abs, max_abs
    else:
        global_vmin = np.min(all_values)
        global_vmax = np.max(all_values)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    axes = axes.flatten()

    visualize_square = False
    if (angle_req is not None and tau_req is not None):
        if (angle_min <= angle_req <= angle_max) and (0.0 <= tau_req <= 1.0):
            visualize_square = True
            values_found = {}

    for i, muscle in enumerate(muscle_names):
        heatmap_data = df.pivot(index='torque_norm', columns='angle', values=i)
        
        X = heatmap_data.columns.values
        Y = heatmap_data.index.values
        Z = heatmap_data.values

        im = axes[i].pcolormesh(X, Y, Z, cmap=cmap, vmin=global_vmin, vmax=global_vmax, shading='auto')

        if visualize_square:
            idx_angle = (np.abs(X - angle_req)).argmin()
            idx_tau = (np.abs(Y - tau_req)).argmin()
            
            closest_angle = X[idx_angle]
            closest_tau = Y[idx_tau]
            values_found[muscle] = Z[idx_tau, idx_angle]

            axes[i].plot(closest_angle, closest_tau, marker='s', color=marker_color, alpha=0.5, markersize=10)
        
        axes[i].set_title(f'{title_prefix}({i}): {muscle}', fontweight='bold')
        axes[i].set_xlabel('Angle')
        axes[i].set_ylabel('Torque Norm')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path)
        print(f"Heatmap saved in: {save_path}")

    plt.show()

    if visualize_square:
        print(f"Valori per (Angle~{closest_angle:.2f}, Torque~{closest_tau:.2f})")
        for muscle, val in values_found.items():
            print(f"{muscle}:    \t {val:.4f}")