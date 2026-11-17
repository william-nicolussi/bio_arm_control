import numpy as np
import pandas as pd
from minjerk_dynamics import generate_trajectory

from src.utils.plotting import plot_stacked

# defaults of near-nes/controller (config/core_models.py, SimulationParams)
RESOLUTION_MS = 1.0
TIME_PREP_MS = 650.0
TIME_MOVE_MS = 500.0
TIME_LOCKED_MS = 150.0
TIME_POST_MS = 200.0   # time_grasp + time_post

PHASES = ("prep", "move", "locked", "post", "hold")
PHASE_COLORS = {"prep": "tab:gray", "move": "tab:green", "locked": "tab:blue",
                "post": "tab:orange", "hold": "tab:purple"}


def phase_intervals(t: np.ndarray, phase: np.ndarray, dt: float) -> list:
    intervals = []
    for name in PHASES:
        t_phase = t[phase == name]
        if len(t_phase):
            intervals.append((name, t_phase[0], t_phase[-1] + dt))
    return intervals


def phase_spans(t: np.ndarray, phase: np.ndarray, dt: float) -> list:
    return [(t_start, t_end, name, PHASE_COLORS[name])
            for name, t_start, t_end in phase_intervals(t, phase, dt)]


class Trajectory:
    def __init__(self, kind: str, name: str, init_angle_deg: float, target_angle_deg: float,
                 resolution_ms: float, data: pd.DataFrame, params: dict = None):
        self.kind = kind                        # "minjerk" or "step"
        self.name = name                        # e.g. "minjerk_500ms"
        self.init_angle_deg = init_angle_deg
        self.target_angle_deg = target_angle_deg
        self.resolution_ms = resolution_ms
        self.data = data                        # t, theta_d, theta_d_dot, theta_d_ddot, phase
        self.params = {} if params is None else params   # parameters specific to the kind

    def __repr__(self) -> str:
        return (f"Trajectory({self.name}: {self.init_angle_deg:g}° → {self.target_angle_deg:g}°, "
                f"{self.duration_ms:g} ms, dt = {self.resolution_ms:g} ms, params={self.params})")

    # ----- constructors -----
    @classmethod
    def minjerk(cls, init_angle_deg: float, target_angle_deg: float,
                resolution_ms: float = RESOLUTION_MS,
                time_prep_ms: float = TIME_PREP_MS, time_move_ms: float = TIME_MOVE_MS,
                time_locked_ms: float = TIME_LOCKED_MS, time_post_ms: float = TIME_POST_MS,
                m1_delay_ms: float = 0.0, hold_in_post: bool = True) -> "Trajectory":
        theta = np.rad2deg(generate_trajectory(
            init_angle_rad=np.deg2rad(init_angle_deg),
            target_angle_rad=np.deg2rad(target_angle_deg),
            resolution_ms=resolution_ms,
            time_prep_ms=time_prep_ms,
            time_move_ms=time_move_ms,
            time_locked_with_feedback_ms=time_locked_ms,
            time_post_ms=time_post_ms,
            m1_delay_ms=m1_delay_ms,
        ))
        # same integer lengths used by generate_trajectory
        n_phases = [int((time_prep_ms - m1_delay_ms) / resolution_ms),
                    int(np.round(time_move_ms / resolution_ms)),
                    int((time_locked_ms + m1_delay_ms) / resolution_ms),
                    int(time_post_ms / resolution_ms)]
        phase = np.repeat(["prep", "move", "locked", "post"], n_phases)
        if len(phase) != len(theta):
            raise RuntimeError(f"{len(theta)} samples from generate_trajectory, "
                               f"{len(phase)} expected from the phase lengths")
        if hold_in_post:
            theta[phase == "post"] = target_angle_deg

        params = dict(time_prep_ms=time_prep_ms, time_move_ms=time_move_ms,
                      time_locked_ms=time_locked_ms, time_post_ms=time_post_ms,
                      m1_delay_ms=m1_delay_ms, hold_in_post=hold_in_post)
        return cls._from_theta(kind="minjerk", name=f"minjerk_{time_move_ms:g}ms",
                               init_angle_deg=init_angle_deg, target_angle_deg=target_angle_deg,
                               theta=theta, phase=phase, resolution_ms=resolution_ms, params=params)

    @classmethod
    def step(cls, init_angle_deg: float, target_angle_deg: float, duration_ms: float,
             resolution_ms: float = RESOLUTION_MS, time_prep_ms: float = 0.0) -> "Trajectory":
        n_total = int(np.round(duration_ms / resolution_ms))
        n_prep = int(np.round(time_prep_ms / resolution_ms))
        if n_prep > n_total:
            raise ValueError(f"time_prep_ms = {time_prep_ms:g} longer than duration_ms = {duration_ms:g}")
        phase = np.repeat(["prep", "hold"], [n_prep, n_total - n_prep])
        theta = np.where(phase == "prep", init_angle_deg, target_angle_deg).astype(float)

        return cls._from_theta(kind="step", name="step",
                               init_angle_deg=init_angle_deg, target_angle_deg=target_angle_deg,
                               theta=theta, phase=phase, resolution_ms=resolution_ms,
                               params=dict(duration_ms=duration_ms, time_prep_ms=time_prep_ms))

    @classmethod
    def _from_theta(cls, kind: str, name: str, init_angle_deg: float, target_angle_deg: float,
                    theta: np.ndarray, phase: np.ndarray, resolution_ms: float,
                    params: dict) -> "Trajectory":
        dt = resolution_ms / 1000.0
        theta_dot = np.gradient(theta, dt)
        data = pd.DataFrame({
            "t": np.arange(len(theta)) * resolution_ms,
            "theta_d": theta,
            "theta_d_dot": theta_dot,
            "theta_d_ddot": np.gradient(theta_dot, dt),
            "phase": phase,
        })
        return cls(kind=kind, name=name, init_angle_deg=init_angle_deg,
                   target_angle_deg=target_angle_deg, resolution_ms=resolution_ms,
                   data=data, params=params)

    # ----- accessors -----
    @property
    def dt_s(self) -> float:
        return self.resolution_ms / 1000.0

    @property
    def duration_ms(self) -> float:
        return len(self.data) * self.resolution_ms

    @property
    def t(self) -> np.ndarray:
        return self.data["t"].to_numpy()

    @property
    def theta_d(self) -> np.ndarray:
        return self.data["theta_d"].to_numpy()

    @property
    def theta_d_dot(self) -> np.ndarray:
        return self.data["theta_d_dot"].to_numpy()

    @property
    def theta_d_ddot(self) -> np.ndarray:
        return self.data["theta_d_ddot"].to_numpy()

    @property
    def phase(self) -> np.ndarray:
        return self.data["phase"].to_numpy()

    def phase_intervals(self) -> list:
        return phase_intervals(self.t, self.phase, self.resolution_ms)

    # ----- plot -----
    def plot(self, title: str = None, show_phases: bool = False,
             save_path: str = None, show: bool = True) -> None:
        plot_stacked(title=title or f"{self.name}: "
                                    f"{self.init_angle_deg:g}° → {self.target_angle_deg:g}°",
                     x=self.t,
                     y=[self.theta_d, self.theta_d_dot, self.theta_d_ddot],
                     y_labels=[r"$\theta_d$ [deg]",
                               r"$\dot{\theta}_d$ [deg/s]",
                               r"$\ddot{\theta}_d$ [deg/s$^2$]"],
                     x_label="Time [ms]",
                     spans=phase_spans(self.t, self.phase, self.resolution_ms) if show_phases else None,
                     save_path=save_path,
                     show=show)