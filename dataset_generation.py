import os
import sys
import time

import numpy as np
import xarray as xr

from src.lut_generator import LUT_generator, mcc
from src.lut import LUT

# --- Parameters ---
W_VALUES = [0.35]  # co-contraction weights
TORQUE_POINTS  = 61 # torque_norm points per angle
EDGE_DENSITY = 0.8 # 0 = uniform grid; closer to 1 = more points near torque_norm 0 and 1

PLOT_HEATMAPS = True

TORQUE_NORM = mcc.torque_norm_grid(TORQUE_POINTS, EDGE_DENSITY)

OUTPUT_DIR = os.path.join("models", "validation_luts")
PARTS_DIR = os.path.join(OUTPUT_DIR, "parts")  # one .nc per W, used to resume an interrupted run
HEATMAPS_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

FILE_NAME = "lut_W_0.35"
NC_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".nc")
CSV_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".csv")

# ----- MAIN ----
def main() -> None:
    todo = [W for W in W_VALUES if not part_is_complete(W)]
    print_parameters(todo)

    if todo:
        # the fast torque function must match the original one before spending time on it
        if mcc.check_torque_fn() > 1e-6:
            print("make_torque_fn differs from total_torque: check the LUT dimensions.")
            sys.exit(1)
        print("\nThis procedure could take a long time.")
        if not confirm():
            print("Aborted.")
            sys.exit(0)
        generate_parts(todo)
    else:
        print("\nAll W values are already generated: only merging.")
        
    # --- Merge and save ---
    merge_parts()
    LUT(NC_PATH).to_csv(CSV_PATH)

    # --- Heatmaps of the W generated in a previous run (the new ones are already saved) ---
    if PLOT_HEATMAPS:
        missing = [W for W in W_VALUES if not os.path.exists(heatmap_path(W))]
        if missing:
            lut = LUT(NC_PATH)
            lut.load_data()
            for W in missing:
                save_heatmap(lut, W)

# ----- GENERATION AND MERGE -----
def generate_parts(todo: list) -> None:
    """Generate one W at a time and save it immediately in PARTS_DIR."""
    start_all = time.perf_counter()
    for i, W in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] W = {float(W):g}")
        start = time.perf_counter()
 
        generator = LUT_generator([W], torque_norm=TORQUE_NORM)
        generator.generate()
        generator.save(part_path(W))  # temporary file + replace: a part on disk is always complete
 
        print(f"W = {float(W):g} completed in {(time.perf_counter() - start) / 60:.1f} min")
        if PLOT_HEATMAPS:  # saved right away: they can be checked while the next W is running
            lut = LUT(part_path(W))
            lut.load_data()
            save_heatmap(lut, W)
    print(f"\nGeneration completed in {(time.perf_counter() - start_all) / 60:.1f} min")
  
def merge_parts() -> None:
    """Merge the parts of the W values in W_VALUES into a single file."""
    parts = [xr.load_dataset(part_path(W)) for W in W_VALUES]
 
    # torque, torque_min and torque_max do not depend on W: they must be the same in every part
    for part in parts[1:]:
        for var in ("torque", "torque_min", "torque_max"):
            if not np.allclose(part[var].values, parts[0][var].values):
                raise ValueError(f"'{var}' differs between parts: check {PARTS_DIR}")
 
    # concatenate only the variables that have W (activation_value); the others are taken once
    comb_lut = xr.concat(parts, dim="W", data_vars="minimal", coords="minimal", compat="override")
    comb_lut = comb_lut.sortby("W")
    print(comb_lut)
 
    generator = LUT_generator(comb_lut.W.values)
    generator.comb_lut = comb_lut  # merged LUT: nothing to generate, only to save
    generator.save(NC_PATH)
  
def heatmap_path(W: float) -> str:
    return os.path.join(HEATMAPS_DIR, f"heatmap_W_{float(W):g}.png")

def save_heatmap(lut: LUT, W: float) -> None:
    """Heatmaps of the 6 activations of one W, saved in HEATMAPS_DIR (not shown)."""
    lut.plot_heatmaps(considered_W=float(W), save_path=heatmap_path(W), show=False)

def part_path(W: float) -> str:
    return os.path.join(PARTS_DIR, f"comb_lut_W_{float(W):g}.nc")
 
def part_is_complete(W: float) -> bool:
    """True if the part of W exists and was generated with the current torque_norm grid."""
    path = part_path(W)
    if not os.path.exists(path):
        return False
    try:
        ds = xr.load_dataset(path)
    except Exception:
        return False  # unreadable file: generate it again
    same_grid = (ds.sizes.get("torque_norm") == len(TORQUE_NORM)
                 and np.allclose(ds["torque_norm"].values, TORQUE_NORM))
    return same_grid and bool(np.isclose(ds["W"].values, W).any())
 
# ----- OTHER FUNCTIONS -----
def print_parameters(todo: list) -> None:
    n_angles = len(mcc.lut.angle)
    n_optimizations = len(todo) * n_angles * (len(TORQUE_NORM) - 2)  # edges are analytic
    done = [W for W in W_VALUES if W not in todo]
    print("LUT dataset generation")
    print(f"  W values      : {W_VALUES}")
    print(f"  Already done  : {done if done else 'none'}")
    print(f"  To generate   : {todo if todo else 'none'}")
    steps = np.diff(TORQUE_NORM)
    print(f"  Torque points : {len(TORQUE_NORM)} per angle (step {steps.min():.4f} at the edges, "
          f"{steps.max():.4f} in the middle)")
    print(f"  Angles        : {n_angles}")
    print(f"  Optimizations : {n_optimizations}")
    print(f"  Parts folder  : {PARTS_DIR}")
    if PLOT_HEATMAPS:
        print(f"  Heatmaps      : {HEATMAPS_DIR}")
    print(f"  Output .nc    : {NC_PATH}")
    print(f"  Output .csv   : {CSV_PATH}")
 
def confirm(question: str = "Proceed?") -> bool:
    """Ask a yes/no question like pip does. Enter alone means yes."""
    while True:
        try:
            answer = input(f"{question} (Y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # no terminal available, or Ctrl+C: treat as "no"
            print()
            return False
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")
 
 
# ----- ENTRY POINT -----
if __name__ == "__main__":
    main()