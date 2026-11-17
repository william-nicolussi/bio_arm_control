import json
import os

import numpy as np
import pandas as pd

from src.sim.elbow_simulation import MUSCLES
from src.sim.trajectory import Trajectory, phase_spans
from src.utils.plotting import plot_cartesian, plot_panels


# ----- names -----
def simulation_path(traj: Trajectory, W: float, K_P: float, K_I: float, K_D: float,
                    delay_ms: float = 0.0, ff: bool = False, root: str = "simulations") -> str:
    name = f"W_{W:g}_KP_{K_P:g}_KI_{K_I:g}_KD_{K_D:g}"
    if delay_ms > 0:
        name += f"_delay_{delay_ms:g}ms"
    if ff and traj.kind == "minjerk":
        name += "_ff"
    folder = f"{traj.name}_{traj.init_angle_deg:g}_to_{traj.target_angle_deg:g}deg"
    return os.path.join(root, folder, name).replace("\\", "/")


def simulation_title(traj: Trajectory, W: float, K_P: float, K_I: float, K_D: float,
                     delay_ms: float = 0.0, ff: bool = False) -> str:
    title = (f"{traj.name}, W={W:g}, {traj.init_angle_deg:g}°→{traj.target_angle_deg:g}°, "
             f"$K_P$={K_P:g}, $K_I$={K_I:g}, $K_D$={K_D:g}")
    if delay_ms > 0:
        title += f", delay={delay_ms:g} ms"
    if ff and traj.kind == "minjerk":
        title += ", ff"
    return title


# ----- save / load -----
def save_simulations(sims: dict, save_path: str, reference: pd.DataFrame, parameters: dict = None) -> None:
    os.makedirs(save_path, exist_ok=True)
    # through JSON, so that the comparison with the saved parameters is fair (tuples -> lists, ...)
    parameters = json.loads(json.dumps(parameters or {}, default=_to_builtin))
    path_json = os.path.join(save_path, "simulations_metadata.json")

    files = {}
    if os.path.exists(path_json):
        with open(path_json) as f:
            old = json.load(f)
        old_params = {k: v for k, v in old["parameters"].items() if k != "allocators"}
        new_params = {k: v for k, v in parameters.items() if k != "allocators"}
        if old_params != new_params:
            raise ValueError(f"{save_path} has simulations with other parameters:\n{old_params}\n"
                             f"New parameters:\n{new_params}")
        files = old["files"]
        parameters["allocators"] = {**old["parameters"].get("allocators", {}),
                                    **parameters.get("allocators", {})}

    for label, df in sims.items():
        files[label] = f"{label}.csv"
        df.to_csv(os.path.join(save_path, files[label]), index=False)
    reference.to_csv(os.path.join(save_path, "angle_ref.csv"), index=False)

    with open(path_json, "w") as f:
        json.dump({"parameters": parameters, "files": files, "reference": "angle_ref.csv",
                   "time_unit": "ms"}, f, indent=2)
    print(f"{len(sims)} simulations saved in: {save_path} ({len(files)} in total)")


def load_simulations(save_path: str, labels: list = None) -> tuple:
    with open(os.path.join(save_path, "simulations_metadata.json")) as f:
        meta = json.load(f)
    files = meta["files"]
    if labels is not None:
        missing = [label for label in labels if label not in files]
        if missing:
            print(f"[warning] not in {save_path}: {missing}. Available: {list(files)}")
        files = {label: files[label] for label in labels if label in files}
    sims = {label: _t_to_ms(pd.read_csv(os.path.join(save_path, name)), meta)
            for label, name in files.items()}
    return sims, meta["parameters"]


def load_reference(save_path: str) -> pd.DataFrame:
    with open(os.path.join(save_path, "simulations_metadata.json")) as f:
        meta = json.load(f)
    return _t_to_ms(pd.read_csv(os.path.join(save_path, meta.get("reference", "angle_ref.csv"))), meta)


# ----- plots -----
def plot_angle(sims: dict, title: str = None, show_phases: bool = False,
               t_min: float = None, t_max: float = None,
               save_dir: str = None, file_name: str = "angle", show: bool = True) -> None:
    sims = {label: _time_window(df, t_min, t_max) for label, df in sims.items()}
    first = next(iter(sims.values()))
    n = len(sims)
    plot_cartesian(title=title or "Elbow angle",
                   x=first["t"].to_numpy(),
                   y=[first["theta_d"].to_numpy()] + [df["theta"].to_numpy() for df in sims.values()],
                   labels=[r"$\theta_d$"] + list(sims),
                   colors=["k"] + [f"C{s}" for s in range(n)],
                   linestyles=["--"] + ["-"] * n,
                   x_label="Time [ms]", y_label="Elbow angle [deg]",
                   spans=_phase_spans(first, show_phases),
                   save_path=_plot_file(save_dir, file_name, t_min, t_max), show=show)


def plot_torque(sims: dict, title: str = None, joint_passive: bool = False, show_phases: bool = False,
                t_min: float = None, t_max: float = None,
                save_dir: str = None, file_name: str = "torque", show: bool = True) -> None:
    sims = {label: _time_window(df, t_min, t_max) for label, df in sims.items()}
    first = next(iter(sims.values()))
    y, labels, colors, linestyles = [], [], [], []
    for s, (label, df) in enumerate(sims.items()):
        curves = [(df["tau_muscle"], label, "-"), (df["tau_req"], None, "--")]
        if joint_passive:
            curves.append((df["tau_muscle"] + df["tau_joint_passive"], None, ":"))
        for curve, curve_label, linestyle in curves:
            y.append(curve.to_numpy())
            labels.append(curve_label)
            colors.append(f"C{s}")
            linestyles.append(linestyle)

    linestyle_legend = [("muscles", "-"), ("requested (ff + PID)", "--")]
    if joint_passive:
        linestyle_legend.append(("muscles + joint damping", ":"))
    plot_cartesian(title=title or "Torque generated by muscle activation",
                   x=first["t"].to_numpy(), y=y, labels=labels, colors=colors, linestyles=linestyles,
                   linestyle_legend=linestyle_legend,
                   x_label="Time [ms]", y_label="Torque [Nm]",
                   spans=_phase_spans(first, show_phases), legend_max_cols=3,
                   save_path=_plot_file(save_dir, file_name, t_min, t_max), show=show)


def plot_activations(sims: dict, title: str = None, state: bool = False, show_phases: bool = False,
                     t_min: float = None, t_max: float = None,
                     save_dir: str = None, file_name: str = None, show: bool = True) -> None:
    prefix = "act" if state else "u"
    sims = {label: _time_window(df, t_min, t_max) for label, df in sims.items()}
    first = next(iter(sims.values()))
    default_title, default_file = (("Muscle activation states", "activation_states") if state
                                   else ("Muscle activations", "activations"))
    plot_panels(title=title or default_title,
                x=first["t"].to_numpy(),
                y=[[df[f"{prefix}_{i}"].to_numpy() for i in range(len(MUSCLES))] for df in sims.values()],
                labels=list(sims), subplot_titles=MUSCLES,
                x_label="Time [ms]", y_label=prefix, ncols=2,
                spans=_phase_spans(first, show_phases),
                save_path=_plot_file(save_dir, file_name or default_file, t_min, t_max), show=show)


# ----- helpers -----
def _time_window(df: pd.DataFrame, t_min: float = None, t_max: float = None) -> pd.DataFrame:
    t = df["t"].to_numpy()
    mask = np.ones(len(df), dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    if not mask.any():
        raise ValueError(f"no data in the time window [{t_min}, {t_max}] ms")
    return df[mask]


def _window_file_name(file_name: str, t_min: float = None, t_max: float = None) -> str:
    if t_min is None and t_max is None:
        return file_name
    lo = "start" if t_min is None else f"{t_min:g}"
    hi = "end" if t_max is None else f"{t_max:g}ms"
    return f"{file_name}_t{lo}-{hi}"


def _plot_file(save_dir: str, file_name: str, t_min: float, t_max: float) -> str:
    if save_dir is None:
        return None
    return os.path.join(save_dir, _window_file_name(file_name, t_min, t_max) + ".png")


def _phase_spans(df: pd.DataFrame, show_phases: bool) -> list:
    if not show_phases or "phase" not in df:
        return None
    t = df["t"].to_numpy()
    return phase_spans(t, df["phase"].to_numpy(), t[1] - t[0])


def _t_to_ms(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # folders saved before the change to ms have no time_unit and t in seconds
    if meta.get("time_unit", "s") == "s":
        df["t"] = df["t"] * 1000.0
    return df


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def experiment_path(traj_name: str, init_angle: float, target_angle: float,
                    W: float, K_P: float, K_I: float, K_D: float,
                    delay_ms: float = 0.0, ff: bool = False, root: str = "simulations") -> str:
    name = f"W_{W:g}_KP_{K_P:g}_KI_{K_I:g}_KD_{K_D:g}"
    if delay_ms > 0:
        name += f"_delay_{delay_ms:g}ms"
    if ff:
        name += "_ff"
    folder = f"{traj_name}_{init_angle:g}_to_{target_angle:g}deg"
    return os.path.join(root, folder, name).replace("\\", "/")


def experiment_title(traj_name: str, init_angle: float, target_angle: float,
                     W: float, K_P: float, K_I: float, K_D: float,
                     delay_ms: float = 0.0, ff: bool = False) -> str:
    title = (f"{traj_name}, W={W:g}, {init_angle:g}°→{target_angle:g}°, "
             f"$K_P$={K_P:g}, $K_I$={K_I:g}, $K_D$={K_D:g}")
    if delay_ms > 0:
        title += f", delay={delay_ms:g} ms"
    if ff:
        title += ", ff"
    return title


def simulation_path(traj: Trajectory, W: float, K_P: float, K_I: float, K_D: float,
                    delay_ms: float = 0.0, ff: bool = False, root: str = "simulations") -> str:
    return experiment_path(traj.name, traj.init_angle_deg, traj.target_angle_deg, W, K_P, K_I, K_D,
                           delay_ms=delay_ms, ff=ff and traj.kind == "minjerk", root=root)


def simulation_title(traj: Trajectory, W: float, K_P: float, K_I: float, K_D: float,
                     delay_ms: float = 0.0, ff: bool = False) -> str:
    return experiment_title(traj.name, traj.init_angle_deg, traj.target_angle_deg, W, K_P, K_I, K_D,
                            delay_ms=delay_ms, ff=ff and traj.kind == "minjerk")

