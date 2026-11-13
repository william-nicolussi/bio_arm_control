import os
import sys

from src.lut.lut_generator import LutGenerator

# --- Generation Parameters ---
P_VALUE = 2               # exponent p of the effort term (1-W) sum u^p
W_VALUES = [0.15]         # co-contraction weights, in [0, 1]
ANGLE_POINTS = 37         # angles over the joint range read from the model
MAX_THETA_DOT = 600.0     # deg/s, symmetric grid [-MAX, MAX]. TODO: max |theta_dot| of the trajectories
VELOCITY_POINTS = 41      # odd, so that 0 is a grid node
TORQUE_POINTS = 61        # torque_norm points per (angle, velocity)
EDGE_DENSITY = 0.8        # 0 = uniform grid; closer to 1 = more points near torque_norm 0 and 1

# --- Output Options ---
SAVE_HEATMAPS = True      # heatmaps of every W at minimum, zero and maximum velocity

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # works from any working directory
MYO_SIM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "myo_sim", "elbow", "myoelbow_1dof6muscles.xml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "luts")
HEATMAPS_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

W_TAG = "_".join(f"{W:g}" for W in W_VALUES)
FILE_NAME = f"lut_p{P_VALUE:g}_W_{W_TAG}"
NC_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".nc")
CSV_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".csv")


# ----- MAIN -----
def main() -> None:
    # the constructor only loads the model and checks W and p: it is fast
    my_lut_generator = LutGenerator(
        path_model=MYO_SIM_MODEL_PATH,
        p_value=P_VALUE,
        w_values=W_VALUES,
    )

    print_parameters(my_lut_generator)
    if not confirm():
        print("Aborted.")
        sys.exit(0)

    my_lut_generator.generate_lut(
        torque_points=TORQUE_POINTS,
        angle_points=ANGLE_POINTS,
        max_theta_dot=MAX_THETA_DOT,
        velocity_points=VELOCITY_POINTS,
        edge_density=EDGE_DENSITY,
    )
    my_lut_generator.check_solver()       # solver vs SLSQP: raises before saving if it is wrong

    my_lut_generator.save(NC_PATH)
    my_lut_generator.to_csv(CSV_PATH)
    if SAVE_HEATMAPS:
        my_lut_generator.save_heatmaps(HEATMAPS_DIR)


# ----- OTHER FUNCTIONS -----
def print_parameters(generator: LutGenerator) -> None:
    n_points = len(W_VALUES) * ANGLE_POINTS * VELOCITY_POINTS * (TORQUE_POINTS - 2)  # edges are analytic
    print("LUT dataset generation")
    print(f"  Model           : {MYO_SIM_MODEL_PATH}")
    print(f"  Muscles         : {generator.muscles}")
    print(f"  W values        : {W_VALUES}")
    print(f"  p               : {P_VALUE:g}")
    print(f"  Angles          : {ANGLE_POINTS} over the joint range of the model")
    print(f"  Velocities      : {VELOCITY_POINTS} from {-MAX_THETA_DOT:g} to {MAX_THETA_DOT:g} deg/s")
    print(f"  Torque points   : {TORQUE_POINTS} per (angle, velocity), edge density {EDGE_DENSITY:g}")
    print(f"  Points to solve : {n_points} ({len(W_VALUES)} W)")
    print(f"  Output .nc      : {NC_PATH}")
    print(f"  Output .csv     : {CSV_PATH}")
    if SAVE_HEATMAPS:
        print(f"  Heatmaps        : {HEATMAPS_DIR}")


def confirm(question: str = "Proceed?") -> bool:
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