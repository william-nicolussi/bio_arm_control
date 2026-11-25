import numpy as np
import pandas as pd
from minjerk_dynamics import generate_trajectory

from src.utils.plotting import plot_images, plot_stacked, plot_xy, values_text
from src.utils.rendering import render_postures, view_camera

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
    
    def __init__(self, kind: str, name: str, theta_start_deg, theta_end_deg, theta_d, phase,
                 resolution_ms: float, params: dict, dof_names: list = None, arm=None, hand=None):
        self.kind = kind                        # "minjerk", "step" or "cartesian"
        self.name = name                        # e.g. "minjerk_500ms"
        self.theta_start_deg = np.atleast_1d(np.asarray(theta_start_deg, dtype=float))
        self.theta_end_deg = np.atleast_1d(np.asarray(theta_end_deg, dtype=float))
        self.resolution_ms = resolution_ms
        self.params = params                    # times of the kind (saved with the simulations)
        self.dof_names = list(dof_names) if dof_names is not None else [f"dof_{j}" for j in range(self.n_dofs)]

        self.theta_d = np.asarray(theta_d, dtype=float).reshape(len(phase), self.n_dofs)
        self.theta_d_dot = np.gradient(self.theta_d, self.dt_s, axis=0)
        self.theta_d_ddot = np.gradient(self.theta_d_dot, self.dt_s, axis=0)
        self.phase = np.asarray(phase)
        self.t = np.arange(len(self.phase)) * resolution_ms

        self.arm = arm
        self.hand_d = None if hand is None else hand[:, :2]
        self.hand_z = None if hand is None else hand[:, 2]
        if hand is not None:
            start, end = arm.hand_position(self.theta_start_deg), arm.hand_position(self.theta_end_deg)
            self.plane_z = start[2]
            self.hand_start, self.hand_end = start[:2], end[:2]
            self.hand_end_z = end[2]

    # ----- constructors -----
    @classmethod
    def minjerk(cls, theta_start_deg, theta_end_deg, resolution_ms: float = RESOLUTION_MS,
                time_prep_ms: float = TIME_PREP_MS, time_move_ms: float = TIME_MOVE_MS,
                time_locked_ms: float = TIME_LOCKED_MS, time_post_ms: float = TIME_POST_MS,
                m1_delay_ms: float = 0.0, hold_in_post: bool = True, dof_names: list = None,
                arm=None) -> "Trajectory":
        """Minimum jerk of near-nes on every angle, same times: the DoFs start and stop together.
        theta_start_deg, theta_end_deg: a number (the elbow) or one value per DoF.
        arm: checks the range of the angles and gives the names of the DoFs; with a hand (planar arm) the
        trajectory has also hand_d, the hand at theta_d."""
        start = np.atleast_1d(np.asarray(theta_start_deg, dtype=float))
        end = np.atleast_1d(np.asarray(theta_end_deg, dtype=float))
        dof_names = _check_arm(arm, start, end, dof_names)
        times = dict(resolution_ms=resolution_ms, time_prep_ms=time_prep_ms, time_move_ms=time_move_ms,
                     time_locked_with_feedback_ms=time_locked_ms, time_post_ms=time_post_ms,
                     m1_delay_ms=m1_delay_ms)
        theta = np.column_stack([np.rad2deg(generate_trajectory(init_angle_rad=np.deg2rad(a),
                                                                target_angle_rad=np.deg2rad(b), **times))
                                 for a, b in zip(start, end)])
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
            theta[phase == "post"] = end

        params = dict(time_prep_ms=time_prep_ms, time_move_ms=time_move_ms,
                      time_locked_ms=time_locked_ms, time_post_ms=time_post_ms,
                      m1_delay_ms=m1_delay_ms, hold_in_post=hold_in_post)
        hand = _hand(arm, theta)
        return cls("minjerk", f"minjerk_{time_move_ms:g}ms", start, end, theta, phase, resolution_ms, params,
                   dof_names, arm=arm if hand is not None else None, hand=hand)

    @classmethod
    def step(cls, theta_start_deg, theta_end_deg, duration_ms: float, resolution_ms: float = RESOLUTION_MS,
             time_prep_ms: float = 0.0, dof_names: list = None, arm=None) -> "Trajectory":
        """Every angle at its start value in the prep phase, then at its end value: a jump (step response of the
        controller). No feedforward: the acceleration of a jump is infinite. arm: as in minjerk."""
        start = np.atleast_1d(np.asarray(theta_start_deg, dtype=float))
        end = np.atleast_1d(np.asarray(theta_end_deg, dtype=float))
        dof_names = _check_arm(arm, start, end, dof_names)
        n_total = int(np.round(duration_ms / resolution_ms))
        n_prep = int(np.round(time_prep_ms / resolution_ms))
        if n_prep > n_total:
            raise ValueError(f"time_prep_ms = {time_prep_ms:g} longer than duration_ms = {duration_ms:g}")
        phase = np.repeat(["prep", "hold"], [n_prep, n_total - n_prep])
        theta = np.where((phase == "prep")[:, None], start, end)
        hand = _hand(arm, theta)
        return cls("step", "step", start, end, theta, phase, resolution_ms,
                   dict(duration_ms=duration_ms, time_prep_ms=time_prep_ms), dof_names,
                   arm=arm if hand is not None else None, hand=hand)

    @classmethod
    def cartesian(cls, arm, theta_start_deg, theta_end_deg, resolution_ms: float = RESOLUTION_MS,
                  time_prep_ms: float = TIME_PREP_MS, time_move_ms: float = TIME_MOVE_MS,
                  time_locked_ms: float = TIME_LOCKED_MS, time_post_ms: float = TIME_POST_MS) -> "Trajectory":
        """Planar arm: angles -> hand (forward kinematics) -> straight line with the minimum-jerk profile of
        near-nes -> angles (inverse kinematics). Only the hand has minimum jerk."""
        if arm.n_dofs != 2 or not arm.has_hand:
            raise ValueError("reaching in the plane needs the planar arm (2 DoFs and the hand)")
        start = np.asarray(theta_start_deg, dtype=float)
        end = np.asarray(theta_end_deg, dtype=float)
        _check_arm(arm, start, end)

        # profile s from 0 to 1, with the phases of near-nes
        s = cls.minjerk(0.0, 1.0, resolution_ms=resolution_ms, time_prep_ms=time_prep_ms,
                        time_move_ms=time_move_ms, time_locked_ms=time_locked_ms, time_post_ms=time_post_ms)
        hand_start = arm.hand_position(start)[:2]
        hand_d = hand_start + s.theta_d * (arm.hand_position(end)[:2] - hand_start)
        theta = inverse_kinematics(arm, hand_d, start, s.t)
        _check_path(arm, theta, s.t)

        params = dict(time_prep_ms=time_prep_ms, time_move_ms=time_move_ms,
                      time_locked_ms=time_locked_ms, time_post_ms=time_post_ms)
        return cls("cartesian", f"cartesian_{time_move_ms:g}ms", start, end, theta, s.phase, resolution_ms,
                   params, arm.dof_names, arm=arm, hand=_hand(arm, theta))

    # ----- save / load -----
    def to_dict(self) -> dict:
        """What is needed to build the trajectory again (from_dict)."""
        return dict(kind=self.kind, theta_start_deg=self.theta_start_deg.tolist(),
                    theta_end_deg=self.theta_end_deg.tolist(), resolution_ms=self.resolution_ms, **self.params)

    @classmethod
    def from_dict(cls, spec: dict, arm=None) -> "Trajectory":
        spec = dict(spec)
        kind = spec.pop("kind")
        if kind == "cartesian":
            return cls.cartesian(arm, **spec)
        return getattr(cls, kind)(arm=arm, **spec)

    # ----- accessors -----
    @property
    def n_dofs(self) -> int:
        return len(self.theta_start_deg)

    @property
    def dt_s(self) -> float:
        return self.resolution_ms / 1000.0

    @property
    def duration_ms(self) -> float:
        return len(self.t) * self.resolution_ms

    @property
    def data(self) -> pd.DataFrame:
        """t, phase and the reference: columns theta_d, ... for 1 DoF (as angle_ref.csv of the elbow),
        theta_d_<dof>, ... for more DoFs."""
        columns = {"t": self.t}
        for name, values in (("theta_d", self.theta_d), ("theta_d_dot", self.theta_d_dot),
                             ("theta_d_ddot", self.theta_d_ddot)):
            for j, dof in enumerate(self.dof_names):
                columns[name if self.n_dofs == 1 else f"{name}_{dof}"] = values[:, j]
        columns["phase"] = self.phase
        return pd.DataFrame(columns)

    @property
    def spans(self) -> list:
        return phase_spans(self.t, self.phase, self.resolution_ms)

    def phase_intervals(self) -> list:
        return phase_intervals(self.t, self.phase, self.resolution_ms)

    @property
    def hand_d_subject(self) -> np.ndarray:
        """Reference of the hand (N, 2): (right, forward) [m] from the shoulder."""
        return self.arm.to_subject(self.hand_d)

    @property
    def distance_m(self) -> float:
        """Distance [m] between the hand at the start and at the end."""
        return float(np.linalg.norm(self.hand_end - self.hand_start))

    def camera(self, view: str):
        """Camera "top", "front" or "side" (rendering.view_camera) looking between the shoulder and the end of
        the hand, at the height of the hand: the same for the views of the trajectory, the player and the video."""
        lookat = [*(self.arm.shoulder_position()[:2] + self.hand_end) / 2, self.plane_z]
        return view_camera(view, lookat, forward_xy=self.arm.subject_axes[1])

    # ----- plot -----
    def plot(self, title: str = None, show_phases: bool = True, save_path: str = None, show: bool = True) -> None:
        """1 DoF: angle, velocity, acceleration. More DoFs: angle and velocity of each DoF, and the hand if any."""
        if self.n_dofs == 1:
            y = [self.theta_d[:, 0], self.theta_d_dot[:, 0], self.theta_d_ddot[:, 0]]
            y_labels = [r"$\theta_d$ [deg]", r"$\dot{\theta}_d$ [deg/s]", r"$\ddot{\theta}_d$ [deg/s$^2$]"]
        else:
            y = [*self.theta_d.T, *self.theta_d_dot.T]
            y_labels = [f"{d} [deg]" for d in self.dof_names] + [f"{d} [deg/s]" for d in self.dof_names]
        if self.hand_d is not None:
            hand = self.hand_d_subject * 100
            y = [hand[:, 0], hand[:, 1]] + y
            y_labels = ["hand right [cm]", "hand forward [cm]"] + y_labels
        plot_stacked(title=title or f"{self.name}: {self.angles_text()}", x=self.t, y=y, y_labels=y_labels,
                     x_label="Time [ms]", spans=self.spans if show_phases else None,
                     save_path=save_path, show=show)

    def plot_hand(self, title: str = None, save_path: str = None, show: bool = True) -> None:
        """Top view of the hand: its path when the angles follow theta_d, the start (square) and the end (circle).
        minjerk: a curve; cartesian: the straight line; step: a jump, the line only joins the two points."""
        hand = self.hand_d_subject * 100
        plot_xy(title=title or f"Hand, top view ({self.name}: {self.angles_text()})",
                curves=[(hand[:, 0], hand[:, 1])], labels=["hand at theta_d"], colors=["k"],
                points=[(*hand[0], "s", "k"), (*hand[-1], "o", "C2")],
                x_label="Right [cm] (from the shoulder)", y_label="Forward [cm] (from the shoulder)",
                save_path=save_path, show=show)

    def plot_views(self, view=("top", "front"), postures: int = 3, model=None, height: int = 480,
                   width: int = 480, quality: str = "normal", title: str = None, save_path: str = None,
                   show: bool = True) -> None:
        """The arm in MuJoCo, from the same views of the player and the video: postures along the move phase
        (the first ones as ghosts, the last one solid) and the path of the hand (a dot every 10 ms, from dark to
        green). model: for the rendering; None = arm.simulation_model(). Needs the hand (planar arm)."""
        if self.hand_d is None:
            raise ValueError("the views need the hand: build the trajectory with arm=arm (planar arm)")
        model = model if model is not None else self.arm.simulation_model(timestep_ms=self.resolution_ms)
        moving = np.flatnonzero(self.phase == "move") if np.any(self.phase == "move") else np.arange(len(self.t))
        rows = np.linspace(moving[0], moving[-1], max(postures, 2)).round().astype(int)
        qpos = np.array([self.arm.full_qpos(self.theta_d[k])[0] for k in rows])
        views = list(view) if isinstance(view, (list, tuple)) else [view]
        images = [render_postures(model, qpos, self.camera(v), height, width, self._path_markers(), quality=quality)
                  for v in views]
        plot_images(title=title or f"{self.name}: {self.angles_text()} (ghosts: earlier postures)", images=images,
                    subplot_titles=views, save_path=save_path, show=show)

    def _path_markers(self) -> list:
        """Spheres for render_postures: the hand every 10 ms from dark to green, the start and the end."""
        every = max(1, int(round(10 / self.resolution_ms)))
        path = np.column_stack([self.hand_d, self.hand_z])[::every]
        green = np.array([0.1, 0.75, 0.2])
        markers = [(p, 0.006, (*(0.1 + (green - 0.1) * c), 1.0)) for p, c in zip(path, np.linspace(0, 1, len(path)))]
        return markers + [((*self.hand_start, self.plane_z), 0.012, (0.05, 0.05, 0.05, 1.0)),
                          ((*self.hand_end, self.hand_end_z), 0.014, (*green, 1.0))]

    def angles_text(self) -> str:
        """e.g. "130° → 0°" or "[90, 0]° → [0, 90]°", for the titles."""
        return f"{values_text(self.theta_start_deg)}° → {values_text(self.theta_end_deg)}°"

    def __repr__(self) -> str:
        hand = f", hand {self.distance_m * 100:.1f} cm" if self.hand_d is not None else ""
        return (f"Trajectory({self.name}: {self.angles_text()}{hand}, {self.duration_ms:g} ms, "
                f"dt = {self.resolution_ms:g} ms, params={self.params})")


# ----- kinematics -----
def inverse_kinematics(arm, hand_d, theta_start_deg, t=None, tol: float = 1e-9) -> np.ndarray:
    """Angles (N, 2) [deg] that put the hand at hand_d (N, 2) [m] (world x, y): Newton with warm start from the
    previous step, finite-difference Jacobian of Arm.hand_position. With the elbow in [0, 130] deg there is one
    posture for each point of the plane. The angles are not checked against the range of the arm (_check_path)."""
    theta = np.asarray(theta_start_deg, dtype=float).copy()
    t = np.arange(len(hand_d)) if t is None else t
    out = np.empty((len(hand_d), 2))
    h = 1e-4   # [deg]
    for k, target in enumerate(hand_d):
        for _ in range(50):
            err = target - arm.hand_position(theta)[:2]
            if np.linalg.norm(err) < tol:
                break
            J = np.column_stack([(arm.hand_position(theta + h * e)[:2] - arm.hand_position(theta - h * e)[:2])
                                 / (2 * h) for e in np.eye(2)])
            theta = theta + np.linalg.solve(J, err)
        else:
            raise RuntimeError(f"inverse kinematics did not converge at t = {t[k]:g} ms")
        out[k] = theta
    return out


def _check_path(arm, theta, t) -> None:
    """The straight line of the hand must stay where the arm can go: every angle inside its range."""
    lo, hi = arm.range_deg.T
    problems = []
    for j, dof in enumerate(arm.dof_names):
        below, above = theta[:, j] < lo[j], theta[:, j] > hi[j]
        if above.any():
            problems.append(f"{dof} up to {theta[:, j].max():.1f} deg (max {hi[j]:.1f}) "
                            f"from t = {t[above][0]:g} ms")
        if below.any():
            problems.append(f"{dof} down to {theta[:, j].min():.1f} deg (min {lo[j]:.1f}) "
                            f"from t = {t[below][0]:g} ms")
    if problems:
        raise ValueError("the straight line of the hand leaves the range of the arm: it needs "
                         + "; ".join(problems) + ". Choose other postures, or the minjerk trajectory (in the angles)")


def _check_arm(arm, start, end, dof_names: list = None) -> list:
    """With an arm: start and end inside its range, and the names of its DoFs."""
    if arm is None:
        return dof_names
    _check_range(arm, start, "start")
    _check_range(arm, end, "end")
    return arm.dof_names


def _hand(arm, theta):
    """World x, y, z (N, 3) [m] of the hand at every row of theta, or None if the arm has no hand."""
    if arm is None or not arm.has_hand:
        return None
    return np.array([arm.hand_position(angles) for angles in theta])


def _check_range(arm, theta, what: str) -> None:
    lo, hi = arm.range_deg.T
    if np.any(theta < lo) or np.any(theta > hi):
        raise ValueError(f"{what} theta = {np.round(theta, 1).tolist()} deg: outside the range of the arm "
                         f"{arm.range_deg.round(1).tolist()} deg")
