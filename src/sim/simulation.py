import itertools
from collections import deque

import mujoco
import numpy as np
import pandas as pd

from src.utils.html_generator import save_player_html
from src.utils.plotting import panel_frames, values_text
from src.utils.rendering import VIEWS, render_frames, side_by_side
from src.utils.video import frames_player, save_video


def full_inertia(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense inertia matrix (nv, nv); the signature of mj_fullM changed in MuJoCo 4."""
    M = np.zeros((model.nv, model.nv))
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, M, data.qM)
    else:
        mujoco.mj_fullM(model, data, M)
    return M


def feedforward_torque(model: mujoco.MjModel, arm, traj) -> np.ndarray:
    """Torque (N, n) [Nm] the muscles must produce to follow theta_d: inverse dynamics projected on the DoFs,
        tau = B^T (M B theta_ddot + qfrc_bias - qfrc_passive)
    (inertia, Coriolis, gravity, joint damping). The constraint forces do no work along B, so they drop out.
    B is constant along the trajectory (linear couplings): no dB/dt term. The passive force of the muscles is
    in tau_0, so the allocator accounts for it. For the elbow (B = 1) this is mj_inverse."""
    data = mujoco.MjData(model)
    tau = np.empty_like(traj.theta_d)
    for k in range(len(traj.t)):
        q, B = arm.full_qpos(traj.theta_d[k])
        data.qpos[:] = q
        data.qvel[:] = B @ np.deg2rad(traj.theta_d_dot[k])
        mujoco.mj_forward(model, data)
        qacc = B @ np.deg2rad(traj.theta_d_ddot[k])
        tau[k] = B.T @ (full_inertia(model, data) @ qacc + data.qfrc_bias - data.qfrc_passive)
    return tau


class Simulation:
   
    def __init__(self, arm, traj, data: pd.DataFrame, params: dict, label: str = "solver"):
        self.arm = arm
        self.traj = traj
        self.data = data
        self.params = params        # W, p, K_P, K_I, K_D, delay_ms, feedforward, settled_start
        self.label = label

    # ----- simulation -----
    @classmethod
    def run(cls, model: mujoco.MjModel, arm, allocator, W: float, traj, K_P, K_I, K_D, delay_ms: float = 0.0,
            feedforward: bool = True, settled_start: bool = True, label: str = "solver",
            verbose: bool = True) -> "Simulation":
        """K_P [Nm/deg], K_I [Nm/(deg s)], K_D [Nm s/deg]: a number (same for every DoF) or one per DoF.
        feedforward: not for a step (infinite acceleration). settled_start: the activation states start at the
        solution of the first step (the arm already holds the start posture); otherwise at 0, and the arm moves
        until they rise. Controlled muscles: ctrl = u, the other muscles of the model: 0."""
        dt = model.opt.timestep
        if not np.isclose(traj.dt_s, dt):
            raise ValueError(f"trajectory resolution = {traj.resolution_ms:g} ms, MuJoCo timestep = {dt * 1000:g} ms")
        if traj.n_dofs != arm.n_dofs:
            raise ValueError(f"the trajectory has {traj.n_dofs} DoFs, the arm {arm.n_dofs}")
        if feedforward and traj.kind == "step":
            raise ValueError("no feedforward for a step: its acceleration is infinite")
        get_activations = allocator.get_activations if hasattr(allocator, "get_activations") else allocator
        n = arm.n_dofs
        K_P, K_I, K_D = (np.broadcast_to(np.asarray(g, dtype=float), (n,)).copy() for g in (K_P, K_I, K_D))
        n_steps = len(traj.t)
        tau_ff = feedforward_torque(model, arm, traj) if feedforward else np.zeros((n_steps, n))

        data = mujoco.MjData(model)
        data.qpos[:], _ = arm.full_qpos(traj.theta_d[0])
        idx = arm.muscle_indices
        if settled_start:
            data.act[idx] = get_activations(W, traj.theta_d[0], np.zeros(n), tau_ff[0])
        mujoco.mj_forward(model, data)
        has_hand = traj.hand_d is not None
        hand_id = model.body("thirdmc").id if has_hand else None

        n_delay = int(round(delay_ms / traj.resolution_ms))
        angle_buf = deque([traj.theta_d[0]] * (n_delay + 1), maxlen=n_delay + 1)   # [0]: n_delay steps ago
        ref_buf = deque([traj.theta_d[0]] * (n_delay + 1), maxlen=n_delay + 1)

        dofs, muscles = arm.dof_names, arm.muscle_names
        per_dof = ["theta", "theta_meas", "theta_dot", "theta_d", "tau_req", "tau_ff", "tau_fb", "tau_I",
                   "tau_muscle", "tau_passive"]
        columns = ([f"{name}_{d}" for name in per_dof for d in dofs] + ["rho", "saturated", "iterations"]
                   + (["hand_x", "hand_y", "hand_z", "hand_d_x", "hand_d_y"] if has_hand else [])
                   + [f"u_{m}" for m in muscles] + [f"act_{m}" for m in muscles])
        log = np.full((n_steps, len(columns)), np.nan)

        e_int, e_prev = np.zeros(n), None
        for k in range(n_steps):
            theta, theta_dot = arm.controlled_state(data.qpos, data.qvel)
            angle_buf.append(theta)
            ref_buf.append(traj.theta_d[k])

            e = ref_buf[0] - angle_buf[0]
            e_int += e * dt
            e_dot = np.zeros(n) if e_prev is None else (e - e_prev) / dt
            e_prev = e
            tau_fb = K_P * e + K_I * e_int + K_D * e_dot
            tau_req = tau_ff[k] + tau_fb

            u = np.asarray(get_activations(W, theta, theta_dot, tau_req), dtype=float).reshape(-1)
            if not np.all(np.isfinite(u)):
                if verbose:
                    print(f"[warning] NaN activations at t = {traj.t[k]:g} ms: simulation stopped")
                break
            data.ctrl[:] = 0.0
            data.ctrl[idx] = u

            _, B = arm.full_qpos(theta)
            info = getattr(allocator, "last", None) or {}
            hand = [*data.xpos[hand_id], *traj.hand_d[k]] if has_hand else []
            log[k] = [*theta, *angle_buf[0], *theta_dot, *traj.theta_d[k], *tau_req, *tau_ff[k], *tau_fb,
                      *(K_I * e_int), *(B.T @ data.qfrc_actuator), *(B.T @ data.qfrc_passive),
                      info.get("rho", np.nan), info.get("saturated", np.nan), info.get("iterations", np.nan),
                      *hand, *u, *data.act[idx]]
            mujoco.mj_step(model, data)

        df = pd.DataFrame(log, columns=columns)
        df.insert(0, "t", traj.t)
        df.insert(1, "phase", traj.phase)
        params = dict(W=float(W), p=getattr(allocator, "p_value", None), K_P=K_P.tolist(), K_I=K_I.tolist(),
                      K_D=K_D.tolist(), delay_ms=float(delay_ms), feedforward=bool(feedforward),
                      settled_start=bool(settled_start))
        return cls(arm, traj, df, params, label)

    # ----- data -----
    @property
    def t(self) -> np.ndarray:
        return self.data["t"].to_numpy()

    @property
    def has_hand(self) -> bool:
        return "hand_x" in self.data

    def dofs(self, name: str) -> np.ndarray:
        """(N, n): the column name_<dof> of every DoF, e.g. dofs("theta")."""
        return self.data[[f"{name}_{d}" for d in self.arm.dof_names]].to_numpy()

    def muscles(self, prefix: str = "u") -> np.ndarray:
        """(N, M): u (allocator) or act (activation state) of every controlled muscle."""
        return self.data[[f"{prefix}_{m}" for m in self.arm.muscle_names]].to_numpy()

    @property
    def hand(self) -> np.ndarray:
        """Hand (N, 2): (right, forward) [m] from the shoulder."""
        return self.arm.to_subject(self.data[["hand_x", "hand_y"]].to_numpy())

    @property
    def hand_d(self) -> np.ndarray:
        """Reference of the hand (N, 2): (right, forward) [m] from the shoulder."""
        return self.arm.to_subject(self.data[["hand_d_x", "hand_d_y"]].to_numpy())

    def hand_error_mm(self) -> np.ndarray:
        """Distance [mm] between hand and reference, in the plane."""
        return 1000 * np.linalg.norm(self.hand - self.hand_d, axis=1)

    def vertical_drift_mm(self) -> np.ndarray:
        """Height of the hand [mm] from its height at the start: how much the plane of the arm is not flat."""
        z = self.data["hand_z"].to_numpy()
        return 1000 * (z - z[0])

    def vertical_error_mm(self) -> np.ndarray:
        """Height of the hand [mm] from the height of the reference (the hand at theta_d)."""
        return 1000 * (self.data["hand_z"].to_numpy() - self.traj.hand_z)

    def hand_speed_cm_s(self, reference: bool = False) -> np.ndarray:
        position = self.hand_d if reference else self.hand
        return 100 * np.linalg.norm(np.gradient(position, self.t / 1000.0, axis=0), axis=1)

    # ----- analysis -----
    def summary(self) -> dict:
        """Errors on the angles (what the PIDs control) and, with a hand, on the hand."""
        valid = self.data.dropna(subset=[f"theta_{d}" for d in self.arm.dof_names])
        row = {}
        moving = valid["phase"].isin(["move", "locked"]).to_numpy()
        for j, d in enumerate(self.arm.dof_names):
            theta = valid[f"theta_{d}"].to_numpy()
            e = valid[f"theta_d_{d}"].to_numpy() - theta
            end, start = self.traj.theta_end_deg[j], self.traj.theta_start_deg[j]
            direction = np.sign(end - start) or 1.0
            row[f"final angle {d} [deg]"] = theta[-1]
            row[f"final error {d} [deg]"] = e[-1]
            row[f"overshoot {d} [deg]"] = max(0.0, np.max(direction * (theta - end))) if end != start else np.nan
            row[f"RMS error {d} [deg]"] = np.sqrt(np.mean(e ** 2))
            row[f"max error {d} [deg]"] = np.abs(e).max()
            if moving.any():
                row[f"RMS error move+locked {d} [deg]"] = np.sqrt(np.mean(e[moving] ** 2))
            row[f"final tau_req {d} [Nm]"] = valid[f"tau_req_{d}"].iloc[-1]
        if self.has_hand:
            error = self.hand_error_mm()
            row["max hand error [mm]"] = np.nanmax(error)
            row["final hand error [mm]"] = error[np.isfinite(error)][-1]
            row["max vertical error [mm]"] = np.nanmax(np.abs(self.vertical_error_mm()))
            row["z excursion [mm]"] = np.nanmax(self.vertical_drift_mm()) - np.nanmin(self.vertical_drift_mm())
        row["rho max"] = self.data["rho"].max()
        row["saturated steps"] = int(np.nansum(self.data["saturated"]))
        row["stopped at [ms]"] = valid["t"].iloc[-1] if len(valid) < len(self.data) else np.nan
        return row

    @property
    def params_title(self) -> str:
        p = self.params
        title = f"W={p['W']:g}"
        if p.get("p") is not None:
            title += f", p={p['p']:g}"
        title += f", $K_P$={values_text(p['K_P'])}, $K_I$={values_text(p['K_I'])}, $K_D$={values_text(p['K_D'])}"
        if p["delay_ms"] > 0:
            title += f", delay={p['delay_ms']:g} ms"
        if p["feedforward"]:
            title += ", ff"
        return title

    @property
    def title(self) -> str:
        return f"{self.traj.name} {self.traj.angles_text()}, {self.params_title}"

    # ----- video (planar arm: needs the hand) -----
    def view_frames(self, model: mujoco.MjModel, step_ms: float = 1000.0 / 120.0, view="top",
                    height: int = 480, width: int = 480, panels: bool = True, show_phases: bool = True,
                    quality: str = "fast") -> tuple:
        """Frames of every view and of the plots, separately: ({name: frames}, t [ms]), one frame every step_ms
        of simulation. Views: the arm with the controlled muscles from light blue (activation 0) to red
        (activation 1), the reference of the hand (black) and the target (green). "plots" (panels=True): the
        joint angles and the hand speed, drawn up to the time of the frame.
        view: "top", "front", "side" (rendering.view_camera), the name of a camera of the model, or an MjvCamera,
        or a list of them. quality: "fast", "normal", "full" (rendering.QUALITY).
        The posture is rebuilt from the logged angles with Arm.full_qpos."""
        if not self.has_hand:
            raise ValueError("video and player need the hand: planar arm (trajectory built with arm=arm)")
        df = self.data.dropna(subset=[f"theta_{d}" for d in self.arm.dof_names])   # the run may stop early
        t = df["t"].to_numpy()
        t_frames = np.arange(t[0], t[-1] + 1e-9, step_ms)
        rows = np.clip(np.searchsorted(t, t_frames), 0, len(t) - 1)

        theta = df[[f"theta_{d}" for d in self.arm.dof_names]].to_numpy()[rows]
        qpos = np.array([self.arm.full_qpos(th)[0] for th in theta])
        act = df[[f"act_{m}" for m in self.arm.muscle_names]].to_numpy()[rows]
        idx = self.arm.muscle_indices
        if np.any(model.actuator_trntype[idx] != mujoco.mjtTrn.mjTRN_TENDON):
            act, tendons = None, None       # no tendon to colour
        else:
            tendons = model.actuator_trnid[idx, 0]

        steps = df.index.to_numpy()[rows]          # rows of the trajectory: the height of the reference is there
        hand_d = np.column_stack([df[["hand_d_x", "hand_d_y"]].to_numpy()[rows], self.traj.hand_z[steps]])
        markers = [(hand_d, 0.010, (0.05, 0.05, 0.05, 1.0)),
                   ((*self.traj.hand_end, self.traj.hand_end_z), 0.014, (0.1, 0.75, 0.2, 0.8))]

        views = list(view) if isinstance(view, (list, tuple)) else [view]
        layers = {}
        for i, v in enumerate(views):
            name = v if isinstance(v, str) else f"camera_{i}"
            layers[name] = render_frames(model, qpos, self._camera(v), height, width, act, tendons, markers, quality)
        if panels:
            layers["plots"] = panel_frames(t_frames, self._video_panels(df), title=self.title,
                                           spans=self.traj.spans if show_phases else None, height_px=height)
        return layers, t_frames

    def video_frames(self, model: mujoco.MjModel, step_ms: float = 1000.0 / 120.0, **kwargs) -> tuple:
        """The frames of view_frames side by side, as one image per time: (frames, t [ms]). kwargs: view,
        height, width, panels, show_phases, quality (see view_frames)."""
        layers, t = self.view_frames(model, step_ms=step_ms, **kwargs)
        return [side_by_side(images) for images in zip(*layers.values())], t

    def save_video(self, model: mujoco.MjModel, path: str, fps: int = 30, slowdown: float = 4.0,
                   hold_s: float = 1.0, **kwargs) -> None:
        """mp4. slowdown: 4 = 4 times slower than the simulation. kwargs: as view_frames."""
        frames, _ = self.video_frames(model, step_ms=1000.0 / fps / slowdown, **kwargs)
        save_video(frames, path, fps=fps, hold_s=hold_s)

    def player(self, model: mujoco.MjModel, step_ms: float = 10.0, fps: int = 30, **kwargs):
        """Player in the notebook with play / pause and a time slider, one frame every step_ms of simulation.
        Last line of a cell to show it (in a loop: display(...)). kwargs: as view_frames."""
        frames, t = self.video_frames(model, step_ms=step_ms, **kwargs)
        return frames_player(frames, t, fps=fps)

    def save_player(self, model: mujoco.MjModel, path: str, step_ms: float = 10.0, fps: int = 30, **kwargs) -> None:
        """The player as a standalone HTML file (open it in a browser, no Python needed), with every view and
        the plots kept apart: in the page, checkboxes show or hide each of them. kwargs: as view_frames."""
        layers, t = self.view_frames(model, step_ms=step_ms, **kwargs)
        save_player_html(layers, path, t, fps=fps, title=f"{self.traj.name} {self.traj.angles_text()}")

    def _camera(self, view):
        """"top", "front", "side" (Trajectory.camera), the name of a camera of the model, or an MjvCamera."""
        return self.traj.camera(view) if isinstance(view, str) and view in VIEWS else view

    def _video_panels(self, df) -> list:
        t = df["t"].to_numpy()
        panels = [dict(title=f"{d} angle", ylabel="[deg]", x=t,
                       curves=[(df[f"theta_d_{d}"].to_numpy(), r"$\theta_d$", "k", "--"),
                               (df[f"theta_{d}"].to_numpy(), "simulation", "C0", "-")])
                  for d in self.arm.dof_names]
        hand = self.arm.to_subject(df[["hand_x", "hand_y"]].to_numpy())
        hand_d = self.arm.to_subject(df[["hand_d_x", "hand_d_y"]].to_numpy())
        speed = [100 * np.linalg.norm(np.gradient(p, t / 1000.0, axis=0), axis=1) for p in (hand_d, hand)]
        panels.append(dict(title="hand speed", ylabel="[cm/s]", x=t,
                           curves=[(speed[0], "reference", "k", "--"), (speed[1], "simulation", "C0", "-")]))
        return panels

    def __repr__(self) -> str:
        return f"Simulation({self.label}: {self.title}, {len(self.data)} steps)"


# ----- PID tuning -----
def pid_grid_search(model: mujoco.MjModel, arm, allocator, W: float, traj, K_P_values, K_I_values, K_D_values,
                    delay_ms: float = 0.0, feedforward: bool = True, settled_start: bool = True,
                    last_ms: float = 200.0) -> pd.DataFrame:
    """Every combination of the gains (each value: a number for every DoF, or one per DoF). Errors on all the
    DoFs together: rms, overshoot (largest of the DoFs), final_err (mean |e| in the last last_ms).
    Sorted by rms; NaN if the run stopped."""
    n_last = int(round(last_ms / traj.resolution_ms))
    rows = []
    combos = list(itertools.product(K_P_values, K_I_values, K_D_values))
    for i, (kp, ki, kd) in enumerate(combos, 1):
        sim = Simulation.run(model, arm, allocator, W, traj, kp, ki, kd, delay_ms=delay_ms,
                             feedforward=feedforward, settled_start=settled_start, verbose=False)
        e = sim.dofs("theta_d") - sim.dofs("theta")
        if np.isnan(e).any():
            rms = overshoot = final_err = np.nan
        else:
            rms = np.sqrt(np.mean(e ** 2))
            direction = np.sign(traj.theta_end_deg - traj.theta_start_deg)
            overshoot = max(0.0, np.max(direction * (sim.dofs("theta") - traj.theta_end_deg)))
            final_err = np.mean(np.abs(e[-n_last:]))
        rows.append(dict(K_P=kp, K_I=ki, K_D=kd, rms=rms, overshoot=overshoot, final_err=final_err))
        print(f"\r{i}/{len(combos)}", end="")
    print()
    return pd.DataFrame(rows).sort_values("rms", na_position="last").reset_index(drop=True)
