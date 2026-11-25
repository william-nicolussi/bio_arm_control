import json
import os

import mujoco
import numpy as np
import pandas as pd

from src.sim.simulation import Simulation
from src.sim.trajectory import Trajectory
from src.utils.plotting import plot_cartesian, plot_panels, plot_xy


# ----- names -----
def simulation_path(root: str, traj_name: str, theta_start_deg, theta_end_deg, W: float, p: float,
                    K_P, K_I, K_D, delay_ms: float = 0.0, feedforward: bool = False) -> str:
    """Folder of an experiment from its parameters: the notebook that simulates and the one that visualizes find
    the same folder. Angles and gains: a number or one per DoF. traj_name: Trajectory.name, e.g. "minjerk_500ms",
    "step", "cartesian_500ms"."""
    folder = f"{traj_name}_{_u(theta_start_deg)}_to_{_u(theta_end_deg)}deg"
    name = f"W_{W:g}_p_{p:g}_KP_{_u(K_P)}_KI_{_u(K_I)}_KD_{_u(K_D)}"
    if delay_ms > 0:
        name += f"_delay_{delay_ms:g}ms"
    if feedforward:
        name += "_ff"
    return os.path.join(root, folder, name).replace("\\", "/")


def experiment_path(sims, root: str) -> str:
    """simulation_path with the trajectory and the parameters of the simulations."""
    first = _first(sims)
    p, traj = first.params, first.traj
    return simulation_path(root, traj.name, traj.theta_start_deg, traj.theta_end_deg, p["W"], p["p"],
                           p["K_P"], p["K_I"], p["K_D"], p["delay_ms"], p["feedforward"])


# ----- save / load -----
def save_simulations(sims, save_path: str, allocators: dict = None) -> None:
    """One CSV per allocator and simulations_metadata.json. allocators: {label: description} (e.g. the path of
    a network). A folder that exists already must have the same parameters: the new allocators are added."""
    sims = _as_dict(sims)
    first = _first(sims)
    parameters = _parameters(first)
    for label, sim in sims.items():
        if _parameters(sim) != parameters:
            raise ValueError(f"{label} has other parameters than {first.label}: one folder, one experiment")
    descriptions = {label: (allocators or {}).get(label, label) for label in sims}

    path_json = os.path.join(save_path, "simulations_metadata.json")
    files = {}
    if os.path.exists(path_json):
        with open(path_json) as f:
            old = json.load(f)
        if old.get("parameters") != parameters:
            raise ValueError(f"{save_path} has simulations with other parameters:\n{old.get('parameters')}\n"
                             f"New parameters:\n{parameters}")
        files = old["files"]
        descriptions = {**old.get("allocators", {}), **descriptions}

    os.makedirs(save_path, exist_ok=True)
    for label, sim in sims.items():
        files[label] = f"{label}.csv"
        sim.data.to_csv(os.path.join(save_path, files[label]), index=False)
    with open(path_json, "w") as f:
        json.dump(dict(parameters=parameters, allocators=descriptions, files=files, mujoco=mujoco.__version__,
                       time_unit="ms"), f, indent=2)
    print(f"{len(sims)} simulations saved in: {save_path} ({len(files)} in total)")


def load_simulations(save_path: str, arm, labels: list = None) -> dict:
    """{label: Simulation} saved with save_simulations. arm: the same Arm of the run (same DoFs and muscles).
    labels: None = all the allocators of the folder."""
    with open(os.path.join(save_path, "simulations_metadata.json")) as f:
        meta = json.load(f)
    parameters = meta["parameters"]
    if parameters["dofs"] != list(arm.dof_names) or parameters["muscles"] != arm.muscle_names:
        raise ValueError(f"{save_path} was saved with another arm: dofs {parameters['dofs']}, "
                         f"muscles {parameters['muscles']}")
    if meta.get("mujoco") != mujoco.__version__:
        print(f"[warning] saved with MuJoCo {meta.get('mujoco')}, now {mujoco.__version__}")
    files = meta["files"]
    if labels is not None:
        missing = [label for label in labels if label not in files]
        if missing:
            print(f"[warning] not in {save_path}: {missing}. Available: {list(files)}")
        files = {label: files[label] for label in labels if label in files}
    traj = Trajectory.from_dict(parameters["trajectory"], arm)
    params = {k: parameters[k] for k in ("W", "p", "K_P", "K_I", "K_D", "delay_ms", "feedforward", "settled_start")}
    return {label: Simulation(arm, traj, pd.read_csv(os.path.join(save_path, name)), dict(params), label)
            for label, name in files.items()}


# ----- summary -----
def summary_table(sims) -> pd.DataFrame:
    """One row per allocator: Simulation.summary."""
    return pd.DataFrame({label: sim.summary() for label, sim in _as_dict(sims).items()}).T


# ----- plots -----
def plot_angles(sims, title: str = None, show_phases: bool = True, t_min: float = None, t_max: float = None,
                save_dir: str = None, show: bool = True) -> None:
    """One panel per DoF: the reference and the angle of every allocator."""
    sims = _as_dict(sims)
    first = _first(sims)
    rows = _window(first, t_min, t_max)
    dofs = first.arm.dof_names
    y = [[first.dofs("theta_d")[rows, j] for j in range(len(dofs))]]
    y += [[sim.dofs("theta")[rows, j] for j in range(len(dofs))] for sim in sims.values()]
    plot_panels(title=title or f"Joint angles ({_title(sims)})", x=first.t[rows], y=y,
                labels=[r"$\theta_d$"] + list(sims), subplot_titles=list(dofs),
                colors=["k"] + [f"C{s}" for s in range(len(sims))], linestyles=["--"] + ["-"] * len(sims),
                x_label="Time [ms]", y_label="Angle [deg]", ncols=min(2, len(dofs)), panel_size=_panel_size(len(dofs)),
                spans=_spans(first, show_phases, t_min, t_max), save_path=_file(save_dir, "angles", t_min, t_max), show=show)


def plot_torques(sims, title: str = None, joint_passive: bool = False, show_phases: bool = True,
                 t_min: float = None, t_max: float = None, save_dir: str = None, show: bool = True) -> None:
    """One panel per DoF: for every allocator the torque of the muscles (solid) and the requested one (dashed);
    joint_passive adds muscles + joint damping (dotted)."""
    sims = _as_dict(sims)
    first = _first(sims)
    rows = _window(first, t_min, t_max)
    dofs = first.arm.dof_names
    y, labels, colors, linestyles = [], [], [], []
    for s, (label, sim) in enumerate(sims.items()):
        curves = [(sim.dofs("tau_muscle"), label, "-"), (sim.dofs("tau_req"), None, "--")]
        if joint_passive:
            curves.append((sim.dofs("tau_muscle") + sim.dofs("tau_passive"), None, ":"))
        for values, curve_label, linestyle in curves:
            y.append([values[rows, j] for j in range(len(dofs))])
            labels.append(curve_label)
            colors.append(f"C{s}")
            linestyles.append(linestyle)
    legend = [("muscles", "-"), ("requested (ff + PID)", "--")]
    if joint_passive:
        legend.append(("muscles + joint damping", ":"))
    plot_panels(title=title or f"Torques ({_title(sims)})", x=first.t[rows], y=y, labels=labels,
                subplot_titles=list(dofs), colors=colors, linestyles=linestyles, linestyle_legend=legend,
                x_label="Time [ms]", y_label="Torque [Nm]", ncols=min(2, len(dofs)), panel_size=_panel_size(len(dofs)),
                spans=_spans(first, show_phases, t_min, t_max), save_path=_file(save_dir, "torques", t_min, t_max), show=show)


def plot_activations(sims, title: str = None, state: bool = False, show_phases: bool = True,
                     t_min: float = None, t_max: float = None, ncols: int = None,
                     save_dir: str = None, show: bool = True) -> None:
    """One panel per muscle: u of every allocator, or (state=True) the activation state of MuJoCo."""
    sims = _as_dict(sims)
    first = _first(sims)
    rows = _window(first, t_min, t_max)
    prefix = "act" if state else "u"
    muscles = first.arm.muscles
    name = "Activation states" if state else "Muscle activations"
    plot_panels(title=title or f"{name} ({_title(sims)})", x=first.t[rows],
                y=[[sim.muscles(prefix)[rows, i] for i in range(len(muscles))] for sim in sims.values()],
                labels=list(sims), subplot_titles=[f"{m.name} ({' + '.join(m.dofs)})" for m in muscles],
                x_label="Time [ms]", y_label=prefix, ncols=ncols or (2 if len(muscles) <= 8 else 4),
                spans=_spans(first, show_phases, t_min, t_max),
                save_path=_file(save_dir, "activation_states" if state else "activations", t_min, t_max), show=show)


def plot_hand_path(sims, title: str = None, save_dir: str = None, show: bool = True) -> None:
    """Top view of the hand: the reference and the hand of every allocator."""
    sims = _as_dict(sims)
    first = _first(sims)
    reference = first.hand_d * 100
    curves = [(reference[:, 0], reference[:, 1])] + [(sim.hand[:, 0] * 100, sim.hand[:, 1] * 100)
                                                      for sim in sims.values()]
    plot_xy(title=title or f"Hand path, top view ({_title(sims)})", curves=curves,
            labels=["reference (hand at theta_d)"] + list(sims),
            colors=["k"] + [f"C{s}" for s in range(len(sims))], linestyles=["--"] + ["-"] * len(sims),
            points=[(*reference[0], "s", "k"), (*reference[-1], "o", "C2")],
            x_label="Right [cm] (from the shoulder)", y_label="Forward [cm] (from the shoulder)",
            save_path=_file(save_dir, "hand_path"), show=show)


def plot_hand_speed(sims, title: str = None, show_phases: bool = True, save_dir: str = None,
                    show: bool = True) -> None:
    sims = _as_dict(sims)
    first = _first(sims)
    plot_cartesian(title=title or f"Hand speed ({_title(sims)})", x=first.t,
                   y=[first.hand_speed_cm_s(reference=True)] + [sim.hand_speed_cm_s() for sim in sims.values()],
                   labels=["reference"] + list(sims), colors=["k"] + [f"C{s}" for s in range(len(sims))],
                   linestyles=["--"] + ["-"] * len(sims), x_label="Time [ms]", y_label="Speed [cm/s]",
                   spans=_spans(first, show_phases), save_path=_file(save_dir, "hand_speed"), show=show)


def plot_hand_error(sims, title: str = None, show_phases: bool = True, save_dir: str = None,
                    show: bool = True) -> None:
    """Distance of the hand from the reference: in the plane (solid) and in height (dotted). The reference is
    the hand at theta_d, with its own height: the plane of the arm is not exactly flat."""
    sims = _as_dict(sims)
    first = _first(sims)
    y, labels, colors, linestyles = [], [], [], []
    for s, (label, sim) in enumerate(sims.items()):
        y += [sim.hand_error_mm(), sim.vertical_error_mm()]
        labels += [label, None]
        colors += [f"C{s}"] * 2
        linestyles += ["-", ":"]
    plot_cartesian(title=title or f"Hand error ({_title(sims)})", x=first.t, y=y, labels=labels, colors=colors,
                   linestyles=linestyles, linestyle_legend=[("error in the plane", "-"),
                                                            ("vertical error (z - z reference)", ":")],
                   x_label="Time [ms]", y_label="[mm]", spans=_spans(first, show_phases),
                   save_path=_file(save_dir, "hand_error"), show=show)


# ----- helpers -----
def _as_dict(sims) -> dict:
    return {sims.label: sims} if isinstance(sims, Simulation) else dict(sims)


def _first(sims) -> Simulation:
    return next(iter(_as_dict(sims).values()))


def _title(sims) -> str:
    return _first(sims).title


def _parameters(sim: Simulation) -> dict:
    """What identifies an experiment, through JSON (tuples -> lists, numpy -> float): fair comparisons."""
    parameters = dict(**sim.params, trajectory=sim.traj.to_dict(), model=sim.arm.path_model,
                      dofs=list(sim.arm.dof_names), muscles=sim.arm.muscle_names)
    return json.loads(json.dumps(parameters, default=_to_builtin))


def _window(sim: Simulation, t_min: float = None, t_max: float = None) -> np.ndarray:
    t = sim.t
    mask = np.ones(len(t), dtype=bool)
    if t_min is not None:
        mask &= t >= t_min
    if t_max is not None:
        mask &= t <= t_max
    if not mask.any():
        raise ValueError(f"no data in the time window [{t_min}, {t_max}] ms")
    return mask


def _spans(sim: Simulation, show_phases: bool, t_min: float = None, t_max: float = None):
    """Phases, cut to the time window (otherwise they would stretch the time axis)."""
    if not show_phases:
        return None
    lo = -np.inf if t_min is None else t_min
    hi = np.inf if t_max is None else t_max
    return [(max(a, lo), min(b, hi), label, color) for a, b, label, color in sim.traj.spans if b > lo and a < hi]


def _panel_size(n_panels: int) -> tuple:
    return (9.0, 4.0) if n_panels == 1 else (5.0, 2.8)


def _file(save_dir: str, name: str, t_min: float = None, t_max: float = None) -> str:
    if save_dir is None:
        return None
    if t_min is not None or t_max is not None:
        lo = "start" if t_min is None else f"{t_min:g}"
        hi = "end" if t_max is None else f"{t_max:g}ms"
        name = f"{name}_t{lo}-{hi}"
    return os.path.join(save_dir, name + ".png")


def _u(values) -> str:
    """20 -> "20", [4.0, 2.0] -> "4_2": angles and gains in a folder name."""
    return "_".join(f"{v:g}" for v in np.atleast_1d(values))


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
