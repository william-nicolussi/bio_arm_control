import itertools
from collections import deque

import mujoco as mj
import numpy as np
import pandas as pd

from src.sim.trajectory import Trajectory

MUSCLES = ["TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"]


def load_model(xml_path: str, timestep_ms: float = 1.0) -> mj.MjModel:
    """MuJoCo model of the elbow; checks that the muscles have the order of the LUT columns."""
    model = mj.MjModel.from_xml_path(xml_path)
    model.opt.timestep = timestep_ms / 1000.0
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != MUSCLES:
        raise ValueError(f"Muscles of the model {names} differ from the LUT order {MUSCLES}")
    return model


def simulate_closed_loop(model: mj.MjModel, allocator, W: float, traj: Trajectory,
                         K_P: float = 0.1, K_I: float = 0.0, K_D: float = 0.0,
                         delay_ms: float = 0.0, feedforward: bool = False,
                         verbose: bool = True) -> pd.DataFrame:
    dt = model.opt.timestep     # [s], for the integral and the derivative of the PID
    if not np.isclose(traj.dt_s, dt):
        raise ValueError(f"trajectory resolution = {traj.resolution_ms:g} ms, "
                         f"MuJoCo timestep = {dt * 1000:g} ms")
    if feedforward and traj.kind != "minjerk":
        raise ValueError(f"feedforward only for minjerk trajectories, not for {traj.kind}")

    theta_d = traj.theta_d
    n_steps = len(theta_d)
    tau_ff = feedforward_torque(model, traj) if feedforward else np.zeros(n_steps)

    data = mj.MjData(model)
    data.qpos[0] = np.deg2rad(traj.init_angle_deg)

    # buffers of the last n_delay + 1 values: [0] is the value of n_delay steps ago
    n_delay = int(round(delay_ms / traj.resolution_ms))
    angle_buf = deque([traj.init_angle_deg] * (n_delay + 1), maxlen=n_delay + 1)
    ref_buf = deque([theta_d[0]] * (n_delay + 1), maxlen=n_delay + 1)

    columns = (["theta", "theta_meas", "theta_dot", "theta_d", "tau_req", "tau_ff", "tau_fb", "tau_I",
                "tau_muscle", "tau_joint_passive"]
               + [f"u_{i}" for i in range(model.nu)] + [f"act_{i}" for i in range(model.nu)])
    log = np.full((n_steps, len(columns)), np.nan)

    e_int, e_prev = 0.0, None
    for k in range(n_steps):
        angle = np.rad2deg(data.qpos[0])
        theta_dot = np.rad2deg(data.qvel[0])
        angle_buf.append(angle)
        ref_buf.append(theta_d[k])

        # PID on the delayed error (positive torque = flexion)
        e = ref_buf[0] - angle_buf[0]
        e_int += e * dt
        e_dot = 0.0 if e_prev is None else (e - e_prev) / dt
        e_prev = e
        tau_fb = K_P * e + K_I * e_int + K_D * e_dot
        tau_req = tau_ff[k] + tau_fb

        # the allocator gets the true state, not the delayed one
        u = np.asarray(allocator(W, angle, theta_dot, tau_req), dtype=float).reshape(-1)
        if not np.all(np.isfinite(u)):
            if verbose:
                print(f"[warning] NaN activations at t = {traj.t[k]:g} ms: simulation stopped")
            break
        data.ctrl[:] = u

        log[k] = [angle, angle_buf[0], theta_dot, theta_d[k], tau_req, tau_ff[k], tau_fb, K_I * e_int,
                  data.qfrc_actuator[0], data.qfrc_passive[0], *u, *data.act]
        mj.mj_step(model, data)

    df = pd.DataFrame(log, columns=columns)
    df.insert(0, "t", traj.t)
    df.insert(1, "phase", traj.phase)
    return df


def feedforward_torque(model: mj.MjModel, traj: Trajectory) -> np.ndarray:
    """Feedforward torque [Nm] by inverse dynamics (mj_inverse) of traj: the torque the muscles must
    produce to follow theta_d (inertia + gravity + joint damping). The passive force of the muscles is
    inside qfrc_actuator, so the allocator accounts for it."""
    data = mj.MjData(model)
    q, qd, qdd = (np.deg2rad(x) for x in (traj.theta_d, traj.theta_d_dot, traj.theta_d_ddot))
    tau = np.empty(len(q))
    for k in range(len(q)):
        data.qpos[0], data.qvel[0], data.qacc[0] = q[k], qd[k], qdd[k]
        mj.mj_inverse(model, data)
        tau[k] = data.qfrc_inverse[0]
    return tau


def pid_grid_search(model: mj.MjModel, allocator, W: float, traj: Trajectory,
                    K_P_values, K_I_values, K_D_values, delay_ms: float = 0.0,
                    feedforward: bool = False, last_ms: float = 200.0) -> pd.DataFrame:
    target = traj.target_angle_deg
    direction = np.sign(target - traj.init_angle_deg) or 1.0
    n_last = int(round(last_ms / traj.resolution_ms))

    rows = []
    combos = list(itertools.product(K_P_values, K_I_values, K_D_values))
    for i, (kp, ki, kd) in enumerate(combos, 1):
        df = simulate_closed_loop(model, allocator, W, traj, K_P=kp, K_I=ki, K_D=kd,
                                  delay_ms=delay_ms, feedforward=feedforward, verbose=False)
        theta = df["theta"].to_numpy()
        e = theta - df["theta_d"].to_numpy()
        if np.isnan(e).any():
            rms = overshoot = final_err = np.nan
        else:
            rms = np.sqrt(np.mean(e ** 2))
            overshoot = max(0.0, np.max(direction * (theta - target)))
            final_err = np.mean(np.abs(e[-n_last:]))
        rows.append(dict(K_P=kp, K_I=ki, K_D=kd, rms=rms, overshoot=overshoot, final_err=final_err))
        print(f"\r{i}/{len(combos)}", end="")
    print()
    return pd.DataFrame(rows).sort_values("rms", na_position="last").reset_index(drop=True)