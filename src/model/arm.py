import warnings
from dataclasses import dataclass

import numpy as np
import mujoco


PLANAR_DOFS = {
    "shoulder": {"elv_angle": 1.0, "shoulder_rot": 1.07},
    "elbow": {"elbow_flexion": 1.0},
}

PLANAR_FIXED_DEG = {"shoulder_elv": 85.7} # horizontal shoulder 85.7

PLANAR_MUSCLES = (
    # shoulder (monoarticular)
    "DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "TMAJ",
    "PECM1", "PECM2", "PECM3", "LAT2", "LAT3", "CORB",   # LAT1: no active force in this workspace
    # biarticular
    "TRIlong", "BIClong", "BICshort",
    # elbow (monoarticular)
    "TRIlat", "TRImed", "ANC", "BRA", "BRD",
)

ELBOW_DOFS = {"elbow": {"r_elbow_flex": 1.0}}

HAND_BODY = "thirdmc"
PLANAR_SUBJECT_AXES = ((-1.0, 0.0), (0.0, -1.0))


@dataclass(frozen=True)
class Muscle:
    """Who the muscle is and what it crosses. The numbers (K, tau_0) are computed by Arm,
    because MuJoCo computes all the muscles together."""
    name: str
    index: int              # actuator id in MuJoCo
    f_max: float            # [N] peak isometric force (F0 of the notes)
    dofs: tuple             # controlled DoFs with a non-zero moment arm in the workspace

    @property
    def is_biarticular(self) -> bool:
        return len(self.dofs) > 1


class Arm:
    def __init__(self, path_model: str, dofs: dict, fixed_deg: dict = None, muscles=None,
                 moment_arm_tol: float = 1e-3, subject_axes=((1.0, 0.0), (0.0, 1.0)), check_paths: bool = True):
        """dofs: {dof name: {MuJoCo joint: coefficient}}, q_joint = fixed + coefficient * c_dof.
        fixed_deg: {MuJoCo joint: value [deg]}. muscles: names of the controlled muscles (None = all);
        the others stay at activation 0, but their passive force is in tau_0.
        moment_arm_tol [m]: a muscle crosses a DoF if its moment arm exceeds it somewhere in the workspace.
        subject_axes: rows = (right, forward) of the subject in world (x, y), for the plots in the plane.
        check_paths: warn if the path of a controlled muscle jumps (see check_muscle_paths)."""
        self.path_model = path_model
        self._model = mujoco.MjModel.from_xml_path(path_model)
        self._data = mujoco.MjData(self._model)
        m = self._model
        if m.na != m.nu:
            raise ValueError("expected one activation state per muscle")
        if m.nq != m.nv:
            raise ValueError("expected only hinge/slide joints: qpos and qvel with the same layout")

        # q = q_fixed + B_user c, then the joint equalities of the model (e.g. the shoulder rhythm)
        self.dof_names = tuple(dofs)
        self._dofs = {dof: dict(joints) for dof, joints in dofs.items()}
        self._q_fixed = np.zeros(m.nq)
        for joint, value in (fixed_deg or {}).items():
            self._q_fixed[self._adr(joint)] = np.deg2rad(value)
        self._B_user = np.zeros((m.nv, self.n_dofs))
        for k, joints in enumerate(dofs.values()):
            for joint, coefficient in joints.items():
                self._B_user[self._adr(joint), k] = coefficient
        self._equalities = self._joint_equalities()

        # muscles
        names = list(muscles) if muscles is not None else [m.actuator(i).name for i in range(m.nu)]
        self._idx = np.array([m.actuator(name).id for name in names])   # KeyError on an unknown name
        self._moment = np.zeros((m.nu, m.nv))
        self.range_deg = self._dof_ranges()
        self.muscles = self._describe_muscles(names, moment_arm_tol)
        self.subject_axes = np.asarray(subject_axes, dtype=float)
        if check_paths:
            jumps = self.check_muscle_paths()
            if jumps:
                warnings.warn("the path of some muscles jumps in the workspace (wrapping): "
                              + "; ".join(f"{j['muscle']} {j['jump_mm']:.0f} mm between {j['theta_from']} and "
                                          f"{j['theta_to']} deg" for j in jumps)
                              + f" (MuJoCo {mujoco.__version__}). Near a jump the torques, and so the activations, jump.")

    @classmethod
    def planar(cls, path_model: str, muscles=PLANAR_MUSCLES):
        """Arm on a table at shoulder height: 2 DoFs (shoulder, elbow)."""
        return cls(path_model, PLANAR_DOFS, PLANAR_FIXED_DEG, muscles, subject_axes=PLANAR_SUBJECT_AXES)

    @classmethod
    def elbow(cls, path_model: str, muscles=None):
        """The elbow alone: 1 DoF, all the muscles of the model (myoelbow_1dof6muscles.xml)."""
        return cls(path_model, ELBOW_DOFS, muscles=muscles)

    # ----- sizes and names -----
    @property
    def n_dofs(self) -> int:
        return len(self.dof_names)

    @property
    def n_muscles(self) -> int:
        return len(self._idx)

    @property
    def muscle_names(self) -> list:
        return [muscle.name for muscle in self.muscles]

    @property
    def has_hand(self) -> bool:
        """The model has the hand (HAND_BODY): the planar arm yes, the elbow model no."""
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY) >= 0

    # ----- from c to the MuJoCo joints -----
    def _adr(self, joint: str) -> int:
        return int(self._model.jnt_qposadr[self._model.joint(joint).id])

    def _joint_equalities(self) -> list:
        """(dependent adr, reference adr or -1, polynomial coefficients) of the active joint equalities.
        MuJoCo: q1 - q1_0 = a0 + a1 (q2 - q2_0) + ... + a4 (q2 - q2_0)^4, with q_0 = qpos0."""
        m = self._model
        user_joints = set(np.flatnonzero(self._B_user.any(axis=1))) | set(np.flatnonzero(self._q_fixed))
        equalities = []
        for i in range(m.neq):
            if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or not m.eq_active0[i]:
                continue
            dependent = int(m.jnt_qposadr[m.eq_obj1id[i]])
            reference = int(m.jnt_qposadr[m.eq_obj2id[i]]) if m.eq_obj2id[i] >= 0 else -1
            if dependent in user_joints:
                raise ValueError(f"joint {m.joint(m.eq_obj1id[i]).name} is set by an equality of the model")
            equalities.append((dependent, reference, m.eq_data[i, :5].copy()))
        return equalities

    def full_qpos(self, theta) -> tuple:
        """theta [deg] (n,) -> qpos of the model [rad] (nq,) and B = dq/dc (nv, n) [rad/rad]."""
        c = np.deg2rad(np.asarray(theta, dtype=float).reshape(self.n_dofs))
        q = self._q_fixed + self._B_user @ c
        B = self._B_user.copy()
        q0 = self._model.qpos0
        for dependent, reference, a in self._equalities:
            if reference < 0:
                q[dependent] = q0[dependent] + a[0]
                B[dependent] = 0.0
                continue
            x = q[reference] - q0[reference]
            q[dependent] = q0[dependent] + a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3 + a[4] * x**4
            B[dependent] = (a[1] + 2 * a[2] * x + 3 * a[3] * x**2 + 4 * a[4] * x**3) * B[reference]
        return q, B

    def full_qvel(self, theta, theta_dot) -> np.ndarray:
        """theta [deg], theta_dot [deg/s] (n,) -> qvel of the model [rad/s] (nv,)."""
        _, B = self.full_qpos(theta)
        return B @ np.deg2rad(np.asarray(theta_dot, dtype=float).reshape(self.n_dofs))

    def _dof_ranges(self) -> np.ndarray:
        """[min, max] [deg] of each DoF: every joint it moves stays inside its limits (the others at 0).
        Exact for linear couplings, like the planar arm."""
        m = self._model
        q, B = self.full_qpos(np.zeros(self.n_dofs))
        ranges = np.tile([-np.inf, np.inf], (self.n_dofs, 1))
        for joint in range(m.njnt):
            if not m.jnt_limited[joint]:
                continue
            adr = m.jnt_qposadr[joint]
            lo, hi = m.jnt_range[joint]
            for k in np.flatnonzero(B[adr]):
                bounds = sorted(((lo - q[adr]) / B[adr, k], (hi - q[adr]) / B[adr, k]))
                ranges[k] = max(ranges[k, 0], bounds[0]), min(ranges[k, 1], bounds[1])
        return np.rad2deg(ranges)

    def _primary_joints(self) -> list:
        """(qpos adr, coefficient) of the first joint of each DoF: the one that measures it."""
        return [(self._adr(next(iter(joints))), next(iter(joints.values()))) for joints in self._dofs.values()]

    def controlled_state(self, qpos, qvel) -> tuple:
        """qpos [rad], qvel [rad/s] of the model -> theta [deg], theta_dot [deg/s] (n,), read on the primary joints."""
        theta = np.array([(qpos[adr] - self._q_fixed[adr]) / a for adr, a in self._primary_joints()])
        theta_dot = np.array([qvel[adr] / a for adr, a in self._primary_joints()])
        return np.rad2deg(theta), np.rad2deg(theta_dot)

    @property
    def muscle_indices(self) -> np.ndarray:
        """Actuator ids of the controlled muscles, in the order of the columns of K."""
        return self._idx.copy()

    def simulation_model(self, timestep_ms: float = 1.0, contacts: bool = False,
                         solref=(0.004, 1.0), solimp=(0.9999, 0.9999, 0.001, 0.5, 2.0),
                         phantom_inertia: float = 1e-5) -> mujoco.MjModel:
        """The model with the couplings as joint equalities, for mj_step: the primary joint of each DoF is free,
        the other joints of a DoF follow it, every other joint (not set by the model) is held at its fixed value.
        No contacts by default: there is no table, and the hand must not hit the body.
        solimp close to 1 (as the equalities of the model itself): almost hard constraints. With the default
        solimp (0.9, 0.95) the weight of the forearm twists the humerus by ~2 deg and the hand drops by ~3 cm.
        Phantom bodies (the OpenSim shoulder chain: 1 g, inertia 1 kg m^2) get phantom_inertia [kg m^2]:
        with 1 kg m^2 they add about 2.8 kg m^2 to the shoulder, ten times the inertia of the arm itself.
        None keeps the model as it is."""
        m = self._model
        spec = mujoco.MjSpec.from_file(self.path_model)
        for equality in spec.equalities:     # the shoulder rhythm of the model: same stiffness as ours
            equality.solref = solref
            equality.solimp = solimp
        if phantom_inertia is not None:
            for body in spec.bodies:
                if body.mass < 0.01 and max(body.inertia) >= 0.1:
                    body.inertia = [phantom_inertia] * 3
        primary = {adr: (k, a) for k, (adr, a) in enumerate(self._primary_joints())}
        dependent = {eq[0] for eq in self._equalities}
        q0 = m.qpos0
        for joint in range(m.njnt):
            adr = int(m.jnt_qposadr[joint])
            if adr in primary or adr in dependent:
                continue
            equality = spec.add_equality()
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.objtype = mujoco.mjtObj.mjOBJ_JOINT
            equality.name1 = m.joint(joint).name
            equality.solref = solref
            equality.solimp = solimp
            dofs = np.flatnonzero(self._B_user[adr])
            if len(dofs) == 0:                               # held at its fixed value
                equality.data[:5] = [self._q_fixed[adr] - q0[adr], 0, 0, 0, 0]
                continue
            k = int(dofs[0])                                 # follows the primary joint of DoF k
            adr_p, a_p = self._primary_joints()[k]
            a1 = self._B_user[adr, k] / a_p                  # q - q_fixed = a1 (q_p - q_fixed_p)
            equality.name2 = m.joint(int(np.flatnonzero(m.jnt_qposadr == adr_p)[0])).name
            a0 = self._q_fixed[adr] - q0[adr] - a1 * (self._q_fixed[adr_p] - q0[adr_p])
            equality.data[:5] = [a0, a1, 0, 0, 0]
        model = spec.compile()
        model.opt.timestep = timestep_ms / 1000.0
        if not contacts:
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        return model

    # ----- MuJoCo -----
    def _forward(self, theta, theta_dot, act_controlled: float) -> np.ndarray:
        """One mj_forward: controlled muscles at act_controlled, the others at 0. Returns B."""
        m, d = self._model, self._data
        q, B = self.full_qpos(theta)
        d.qpos[:] = q
        d.qvel[:] = B @ np.deg2rad(np.asarray(theta_dot, dtype=float).reshape(self.n_dofs))
        d.act[:] = 0.0
        d.act[self._idx] = act_controlled
        mujoco.mj_forward(m, d)   # computes the state, does not advance in time
        return B

    def _moment_c(self, B: np.ndarray) -> np.ndarray:
        """dL/dc (nu, n) [m/rad] at the current state: MuJoCo moment (dL/dq) times B."""
        d = self._data
        if hasattr(d, "moment_rownnz"):    # MuJoCo >= 3.2: sparse moment
            mujoco.mju_sparse2dense(self._moment, d.actuator_moment, d.moment_rownnz,
                                    d.moment_rowadr, d.moment_colind)
        else:                              # older MuJoCo: dense (nu, nv)
            self._moment[:] = np.asarray(d.actuator_moment).reshape(self._moment.shape)
        return self._moment @ B

    # ----- the linear map of the notes -----
    def get_tau0(self, theta) -> np.ndarray:
        """tau_0(theta) (n,) [Nm]: torque with all activations at 0 (passive forces of ALL the muscles).
        The passive force depends only on the muscle length, so only on the angles: theta_dot = 0.
        qfrc_actuator contains only the muscles: no gravity, no joint damping."""
        B = self._forward(theta, np.zeros(self.n_dofs), 0.0)
        return B.T @ self._data.qfrc_actuator

    def get_K(self, theta, theta_dot) -> np.ndarray:
        """K (n, M) [Nm]: column i = torque of muscle i at activation 1, minus its passive part.
        Two mj_forward instead of M + 1: the force of each muscle depends only on its own activation,
        and MuJoCo gives it muscle by muscle (actuator_force); torque = moment arm x force."""
        _, K = self.get_tau0_K(theta, theta_dot)
        return K

    def get_tau0_K(self, theta, theta_dot) -> tuple:
        """tau_0 (n,) and K (n, M) with two mj_forward."""
        B = self._forward(theta, theta_dot, 0.0)
        tau_0 = B.T @ self._data.qfrc_actuator      # passive: independent of theta_dot
        force_0 = self._data.actuator_force.copy()
        dL_dc = self._moment_c(B)

        self._forward(theta, theta_dot, 1.0)
        delta_force = self._data.actuator_force - force_0   # active force at u = 1, per muscle
        K = dL_dc[self._idx].T * delta_force[self._idx]     # qfrc = (dL/dc)^T force
        return tau_0, K

    def get_tau(self, tau_0, K, u) -> np.ndarray:
        """tau(u) = tau_0 + K u: the linear model of the notes. No MuJoCo here."""
        return tau_0 + K @ u

    def get_moment_arms(self, theta) -> np.ndarray:
        """Moment arms (n, M) [m] of the controlled muscles: r = -dL/dc,
        positive if the muscle pulls the DoF in the positive direction."""
        B = self._forward(theta, np.zeros(self.n_dofs), 0.0)
        return -self._moment_c(B)[self._idx].T

    def hand_position(self, theta, body: str = HAND_BODY) -> np.ndarray:
        """Position [m] (3,) of a body of the hand: only the kinematics, no forces."""
        self._data.qpos[:], _ = self.full_qpos(theta)
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xpos[self._model.body(body).id].copy()

    def shoulder_position(self) -> np.ndarray:
        """Position [m] (3,) of the shoulder (origin of the humerus) in the planar posture."""
        self._data.qpos[:], _ = self.full_qpos(np.zeros(self.n_dofs))
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xpos[self._model.body("humerus").id].copy()

    def to_subject(self, xy) -> np.ndarray:
        """World x, y [m] (..., 2) -> (right, forward) [m] from the shoulder, in the frame of the subject."""
        return (np.asarray(xy, dtype=float) - self.shoulder_position()[:2]) @ self.subject_axes.T

    # ----- muscles -----
    def check_muscle_paths(self, step_deg: float = 5.0, tol_m: float = 0.005) -> list:
        """Jumps of the length of the controlled muscles on a grid of the workspace: between two neighbouring
        points the length must change as the moment arms predict (trapezoid); a difference above tol_m is a jump
        of the path (a wrapping that switches side). Returns the worst jump of each muscle."""
        axes = [np.arange(lo, hi + 1e-9, step_deg) for lo, hi in self.range_deg]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        shape = grid.shape[:-1]
        L = np.empty(shape + (self.n_muscles,))
        dL = np.empty(shape + (self.n_muscles, self.n_dofs))
        for index in np.ndindex(shape):
            B = self._forward(grid[index], np.zeros(self.n_dofs), 0.0)
            L[index] = self._data.actuator_length[self._idx]
            dL[index] = self._moment_c(B)[self._idx]
        worst = {}
        for k in range(self.n_dofs):
            change = np.diff(L, axis=k)
            predicted = 0.5 * (np.take(dL[..., k], range(1, shape[k]), axis=k)
                               + np.take(dL[..., k], range(shape[k] - 1), axis=k)) * np.deg2rad(step_deg)
            mismatch = np.abs(change - predicted)
            for index in zip(*np.nonzero(mismatch > tol_m)):
                *point, muscle = index
                if mismatch[index] > worst.get(muscle, {}).get("jump_m", 0.0):
                    after = list(point)
                    after[k] += 1
                    worst[muscle] = dict(muscle=self.muscles[muscle].name, jump_m=mismatch[index],
                                         jump_mm=1000 * mismatch[index],
                                         theta_from=grid[tuple(point)].round(1).tolist(),
                                         theta_to=grid[tuple(after)].round(1).tolist())
        return list(worst.values())

    def workspace_grid(self, points: int = 5) -> np.ndarray:
        """Uniform grid (points^n, n) [deg] on the ranges of the DoFs."""
        axes = [np.linspace(lo, hi, points) for lo, hi in self.range_deg]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.n_dofs)

    def _describe_muscles(self, names: list, tol: float) -> list:
        m = self._model
        grid = self.workspace_grid()
        r_max = np.max([np.abs(self.get_moment_arms(theta)) for theta in grid], axis=0)
        useless = [name for name, crosses in zip(names, (r_max > tol).any(axis=0)) if not crosses]
        if useless:
            raise ValueError(f"zero moment arm on every DoF (they never change the torque): {useless}")
        # a moment arm is not enough: the active force can be 0 (fibres outside the force-length range)
        K_max = np.max([np.abs(self.get_K(theta, np.zeros(self.n_dofs))) for theta in grid], axis=0)
        no_force = [name for name, k in zip(names, K_max.max(axis=0)) if k < 1e-6]
        if no_force:
            raise ValueError(f"no active torque anywhere in the workspace (fibres outside the force-length range): "
                             f"{no_force}")
        muscles = []
        for k, (name, i) in enumerate(zip(names, self._idx)):
            is_muscle = m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
            muscles.append(Muscle(name=name, index=int(i),
                                  f_max=float(m.actuator_gainprm[i, 2]) if is_muscle else np.nan,
                                  dofs=tuple(dof for j, dof in enumerate(self.dof_names) if r_max[j, k] > tol)))
        return muscles

    def __repr__(self) -> str:
        ranges = ", ".join(f"{dof} [{lo:g}, {hi:g}] deg" for dof, (lo, hi) in zip(self.dof_names, self.range_deg))
        return f"Arm({self.n_dofs} DoFs: {ranges}; {self.n_muscles} muscles)"
