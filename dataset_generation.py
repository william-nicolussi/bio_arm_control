import os
import sys

from src.lut.lut_generator import LutGenerator

# --- Generation Parameters ---
P_VALUE = 3 # exponent p of the effort term (1-W) sum u^p
W_VALUES = [0.0, 0.000201396, 0.00228193, 0.0110775, 0.0369482, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # co-contraction weights, in [0, 1]
ANGLE_POINTS = [0.0, 3.421, 6.842, 10.263, 13.684, 17.105, 20.526, 23.947, 27.368, 30.79, 34.211, 37.632, 41.053, 44.474, 47.895, 51.316, 54.737, 58.158, 61.579, 65.0, 68.421, 71.842, 75.263, 78.684, 82.105, 85.526, 88.947, 92.369, 95.79, 99.211, 100.677, 102.143, 103.609, 105.075, 106.541, 108.008, 109.474, 110.94, 112.406, 113.872, 115.338, 116.805, 118.271, 119.737, 121.203, 122.669, 124.135, 125.602, 127.068, 128.534, 130.0] # angles over the joint range read from the model
MAX_THETA_DOT = 600.0     # deg/s, symmetric grid [-MAX, MAX]
VELOCITY_POINTS = 23 # odd, so that 0 is a grid node
TORQUE_POINTS = 74 # torque_norm points per (angle, velocity)
EDGE_DENSITY = 0.8        # 0 = uniform grid; closer to 1 = more points near torque_norm 0 and 1

FILE_NAME = "lut_p3_W_all"

# --- Output Options ---
SAVE_PLOTS = False
CONSIDERED_VEL = [-MAX_THETA_DOT, 0, MAX_THETA_DOT]

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # works from any working directory
MYO_SIM_MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "myo_sim", "elbow", "myoelbow_1dof6muscles.xml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "luts", "W_all", "p"+f"{P_VALUE}")
PLOTS_DIR = os.path.join(OUTPUT_DIR, FILE_NAME + "_plots")

NC_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".nc")
#CSV_PATH = os.path.join(OUTPUT_DIR, FILE_NAME + ".csv")


# ----- MAIN -----
def main() -> None:
    # the constructor only loads the model and checks W and p: it is fast
    my_lut_generator = LutGenerator(
        path_model=MYO_SIM_MODEL_PATH,
        p_value=P_VALUE,
        w_values=W_VALUES,
    )

    #print_parameters(my_lut_generator)
    if not confirm():
        print("Aborted.")
        sys.exit(0)

    my_lut_generator.generate_lut(
        lut_name=FILE_NAME,
        torque_points=TORQUE_POINTS,
        angle_points=ANGLE_POINTS,
        max_theta_dot=MAX_THETA_DOT,
        velocity_points=VELOCITY_POINTS,
        edge_density=EDGE_DENSITY,
    )
    my_lut_generator.check_solver()

    my_lut_generator.lut.save_nc(OUTPUT_DIR)
    #my_lut_generator.lut.save_csv(OUTPUT_DIR)

    if SAVE_PLOTS:
        my_lut_generator.lut.plot_tau_0(save_path=f"{PLOTS_DIR}/tau_0.png",
                                        show=False)
        my_lut_generator.lut.plot_heatmaps_k(save_path=f"{PLOTS_DIR}/hmps_k.png",
                                            show=False)
        my_lut_generator.lut.plot_cost_all_w(save_path=f"{PLOTS_DIR}/cost_all_w.png",
                                             show=False)
        for vel in CONSIDERED_VEL:
            my_lut_generator.lut.plot_tau_min_max(velocity=vel,
                                            save_path=f"{PLOTS_DIR}/tau_min_max_v_{vel:g}.png",
                                            show=False)
            my_lut_generator.lut.plot_tau_with_tau_min_max(velocity=vel,
                                                    save_path=f"{PLOTS_DIR}/tau_v_{vel:g}.png",
                                                    show=False)
        
        for w in W_VALUES:
            my_lut_generator.lut.plot_heatmaps_activations(w=w,
                                                           save_path=f"{PLOTS_DIR}/hmps_acts_{w:g}.png",
                                                           show=False)
            my_lut_generator.lut.plot_cost(w=w,
                                           save_path=f"{PLOTS_DIR}/cost_{w:g}.png",
                                           show=False)
            my_lut_generator.lut.plot_heatmap_cost(w=w,
                                                   save_path=f"{PLOTS_DIR}/hmp_cost_{w:g}.png",
                                                   show=False)

# ----- OTHER FUNCTIONS -----
def print_parameters(generator: LutGenerator) -> None:
    n_points = len(W_VALUES) * ANGLE_POINTS * VELOCITY_POINTS * (TORQUE_POINTS - 2)  # edges are analytic
    print("LUT dataset generation")
    print(f"  File name       : {FILE_NAME}")
    print(f"  Model           : {MYO_SIM_MODEL_PATH}")
    print(f"  Muscles         : {generator.muscles}")
    print(f"  W values        : {W_VALUES}")
    print(f"  p               : {P_VALUE:g}")
    print(f"  Angles          : {ANGLE_POINTS} over the joint range of the model")
    print(f"  Velocities      : {VELOCITY_POINTS} from {-MAX_THETA_DOT:g} to {MAX_THETA_DOT:g} deg/s")
    print(f"  Torque points   : {TORQUE_POINTS} per (angle, velocity), edge density {EDGE_DENSITY:g}")
    print(f"  Points to solve : {n_points} ({len(W_VALUES)} W)")
    print(f"  Output .nc      : {NC_PATH}")
    #print(f"  Output .csv     : {CSV_PATH}")
    if SAVE_PLOTS:
        print(f"  Plots           : {PLOTS_DIR}")


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