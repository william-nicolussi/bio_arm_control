import os
import sys
import time

import numpy as np
import xarray as xr

from src.lut_generator import LUT_generator, mcc
from src.lut import LUT

# --- Parameters ---
W_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # co-contraction weights
TORQUE_STEPS = 30            # torque intervals per angle (grid has TORQUE_STEPS + 1 points)

OUTPUT_DIR = os.path.join("models", "W_all")
PARTS_DIR = os.path.join(OUTPUT_DIR, "parts")  # one .nc per W, used to resume an interrupted run

FILE_NAME = "lut_W_all"
NC_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".nc")
CSV_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".csv")


# ----- MAIN ----
def main() -> None:
    todo = [W for W in W_VALUES if not part_is_complete(W)]
    print_parameters(todo)

    if todo:
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


# ----- GENERATION AND MERGE -----
def generate_parts(todo: list) -> None:
    """Generate one W at a time and save it immediately in PARTS_DIR."""
    start_all = time.perf_counter()
    for i, W in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] W = {float(W):g}")
        start = time.perf_counter()
 
        generator = LUT_generator([W], torque_steps=TORQUE_STEPS)
        generator.generate()
        generator.save(part_path(W))  # temporary file + replace: a part on disk is always complete
 
        print(f"W = {float(W):g} completed in {(time.perf_counter() - start) / 60:.1f} min")
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
 
    LUT(NC_PATH).save(comb_lut)
 
 
def part_path(W: float) -> str:
    return os.path.join(PARTS_DIR, f"comb_lut_W_{float(W):g}.nc")
 
 
def part_is_complete(W: float) -> bool:
    """True if the part of W exists and was generated with the current TORQUE_STEPS."""
    path = part_path(W)
    if not os.path.exists(path):
        return False
    try:
        ds = xr.load_dataset(path)
    except Exception:
        return False  # unreadable file: generate it again
    return ds.sizes.get("torque_norm") == TORQUE_STEPS + 1 and bool(np.isclose(ds["W"].values, W).any())
 
 
# ----- OTHER FUNCTIONS -----
def print_parameters(todo: list) -> None:
    n_angles = len(mcc.lut.angle)
    n_optimizations = len(todo) * n_angles * (TORQUE_STEPS + 1)
    done = [W for W in W_VALUES if W not in todo]
    print("LUT dataset generation")
    print(f"  W values      : {W_VALUES}")
    print(f"  Already done  : {done if done else 'none'}")
    print(f"  To generate   : {todo if todo else 'none'}")
    print(f"  Torque steps  : {TORQUE_STEPS} ({TORQUE_STEPS + 1} points per angle)")
    print(f"  Angles        : {n_angles}")
    print(f"  Optimizations : {n_optimizations}")
    print(f"  Parts folder  : {PARTS_DIR}")
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