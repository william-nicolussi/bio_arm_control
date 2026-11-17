import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="lut_nn")
class NnPreprocessing(keras.layers.Layer):

    def __init__(self, torque_min, torque_max, angles=None, velocities=None,
                 angle_min=None, angle_max=None, velocity_min=None, velocity_max=None, **kwargs):
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)

        # tables (n_angles, n_velocities)
        self.torque_min = np.asarray(torque_min, dtype="float32")
        self.torque_max = np.asarray(torque_max, dtype="float32")
        if self.torque_min.ndim != 2 or self.torque_min.shape != self.torque_max.shape:
            raise ValueError("torque_min / torque_max must be 2D (n_angles, n_velocities) with the same shape")
        n_angles, n_velocities = self.torque_min.shape

        # grid nodes; models saved before non-uniform grids have only min and max: uniform grid, as before
        if angles is None:
            angles = np.linspace(angle_min, angle_max, n_angles)
        if velocities is None:
            velocities = np.linspace(velocity_min, velocity_max, n_velocities)
        self.angles = self._check_nodes("angle", angles, n_angles)
        self.velocities = self._check_nodes("velocity", velocities, n_velocities)

        self.angle_min, self.angle_max = float(self.angles[0]), float(self.angles[-1])
        self.velocity_min, self.velocity_max = float(self.velocities[0]), float(self.velocities[-1])
        self.n_angles, self.n_velocities = n_angles, n_velocities

    @staticmethod
    def _check_nodes(name: str, nodes, n: int) -> np.ndarray:
        nodes = np.asarray(nodes, dtype="float32").reshape(-1)
        if len(nodes) != n or n < 2 or np.any(np.diff(nodes) <= 0):
            raise ValueError(f"The {name} grid must be strictly increasing with {n} values (one per table row/column)")
        return nodes

    @classmethod
    def from_lut(cls, lut, **kwargs):
        expected = (len(lut.theta), len(lut.theta_dot))
        if lut.tau_min.shape != expected or lut.tau_max.shape != expected:
            raise ValueError(f"tau_min / tau_max must have shape {expected}")
        return cls(torque_min=lut.tau_min, torque_max=lut.tau_max,
                   angles=lut.theta, velocities=lut.theta_dot, **kwargs)

    @staticmethod
    def _cell(x, nodes):
        """Index of the grid cell (0 ... n-2) and position inside it (0 ... 1), for any increasing grid:
        the index is the number of inner nodes <= x. x (batch, 1), already clipped to the grid."""
        inner = ops.convert_to_tensor(nodes[1:-1])                           # (n-2,)
        idx = ops.sum(ops.cast(x >= inner, "int32"), axis=-1, keepdims=True)  # (batch, 1)
        t = ops.convert_to_tensor(nodes)
        x0, x1 = ops.take(t, idx), ops.take(t, idx + 1)
        return idx, (x - x0) / (x1 - x0)

    def _bilinear(self, table, ia, fa, iv, fv):
        # table (n_angles, n_velocities) read as a flat vector: element [a, v] is at a * n_velocities + v
        t = ops.convert_to_tensor(table.reshape(-1))
        nv = self.n_velocities
        i00 = ia * nv + iv
        t00, t01 = ops.take(t, i00), ops.take(t, i00 + 1)
        t10, t11 = ops.take(t, i00 + nv), ops.take(t, i00 + nv + 1)
        return (1.0 - fa) * ((1.0 - fv) * t00 + fv * t01) + fa * ((1.0 - fv) * t10 + fv * t11)

    def call(self, inputs):
        W, angle, velocity, torque = inputs   # each (batch, 1)

        # saturation: outside the grid the LUT has no data
        angle = ops.clip(angle, self.angle_min, self.angle_max)
        velocity = ops.clip(velocity, self.velocity_min, self.velocity_max)

        # torque limits at (angle, velocity), interpolated like Lut.get_activations
        ia, fa = self._cell(angle, self.angles)
        iv, fv = self._cell(velocity, self.velocities)
        t_min = self._bilinear(self.torque_min, ia, fa, iv, fv)
        t_max = self._bilinear(self.torque_max, ia, fa, iv, fv)

        torque_norm_raw = (torque - t_min) / ops.maximum(t_max - t_min, 1e-6)
        features = ops.concatenate([
            ops.clip(W, 0.0, 1.0),
            (angle - self.angle_min) / (self.angle_max - self.angle_min),
            (velocity - self.velocity_min) / (self.velocity_max - self.velocity_min),
            ops.clip(torque_norm_raw, 0.0, 1.0),
        ], axis=-1)
        return features, torque_norm_raw

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        return (batch, 4), (batch, 1)

    def get_config(self):
        config = super().get_config()
        config.pop("trainable", None)
        config.update(
            angles=self.angles.tolist(), velocities=self.velocities.tolist(),
            torque_min=self.torque_min.tolist(), torque_max=self.torque_max.tolist(),
        )
        return config