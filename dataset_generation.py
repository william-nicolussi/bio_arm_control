import os
import sys
import time

from src.lut_generator import LUT_generator, mcc
from src.lut import LUT

# --- Parameters ---
W_VALUES = [0.2]  # co-contraction weights
TORQUE_STEPS = 2            # torque intervals per angle (grid has TORQUE_STEPS + 1 points)

OUTPUT_DIR = os.path.join("models", "W_all")
FILE_NAME = "comb_lut_W_all"
NC_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".nc")
CSV_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".csv")


# ----- MAIN ----
def main() -> None:
    print_parameters()
    print("\nThis procedure could take a long time.")
    if not confirm():
        print("Aborted.")
        sys.exit(0)
 
    # --- Generation ---
    start = time.perf_counter()
    my_generator = LUT_generator(W_VALUES, torque_steps=TORQUE_STEPS)
    comb_lut = my_generator.generate()
    elapsed = time.perf_counter() - start
    print(f"\nGeneration completed in {elapsed / 60:.1f} min")
    print(comb_lut)

    # --- Saving ---
    # single file with all W values (save() also creates OUTPUT_DIR)
    my_generator.save(NC_PATH)
    LUT(NC_PATH).to_csv(CSV_PATH)

# ----- OTHER FUNCTIONS -----
def print_parameters() -> None:
    n_angles = len(mcc.lut.angle)
    n_optimizations = len(W_VALUES) * n_angles * (TORQUE_STEPS + 1)
    print("LUT dataset generation")
    print(f"  W values      : {W_VALUES}")
    print(f"  Torque steps  : {TORQUE_STEPS} ({TORQUE_STEPS + 1} points per angle)")
    print(f"  Angles        : {n_angles}")
    print(f"  Optimizations : {n_optimizations}")
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