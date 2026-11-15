import keras
import numpy as np
from keras import ops

@keras.saving.register_keras_serializable(package="lut_nn")
class NnPreprocessing(keras.layers.Layer):

    def __init__(self, angle_min, angle_max, velocity_min, velocity_max,
                torque_min, torque_max, **kwargs):
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.velocity_min = float(velocity_min)
        self.velocity_max = float(velocity_max)

        # tables (n_angles, n_velocities)
        self.torque_min = np.asarray(torque_min, dtype="float32")
        self.torque_max = np.asarray(torque_max, dtype="float32")
        if self.torque_min.ndim != 2 or self.torque_min.shape != self.torque_max.shape:
            raise ValueError("torque_min / torque_max must be 2D (n_angles, n_velocities) with the same shape")

        self.n_angles, self.n_velocities = self.torque_min.shape
        self.angle_step = (self.angle_max - self.angle_min) / (self.n_angles - 1)
        self.velocity_step = (self.velocity_max - self.velocity_min) / (self.n_velocities - 1)
        
    @classmethod
    def from_lut(cls, lut, **kwargs):
        # the layer finds the cell as (x - x_min) / step: valid only on uniform grids
        for name, axis in (("angle", lut.theta), ("velocity", lut.theta_dot)):
            steps = np.diff(axis)
            if len(axis) < 2 or np.any(steps <= 0) or not np.allclose(steps, steps[0]):
                raise ValueError(f"The {name} grid of the LUT must be uniform and increasing")

        expected = (len(lut.theta), len(lut.theta_dot))
        if lut.tau_min.shape != expected or lut.tau_max.shape != expected:
            raise ValueError(f"tau_min / tau_max must have shape {expected}")

        return cls(angle_min=lut.theta[0], angle_max=lut.theta[-1],
                velocity_min=lut.theta_dot[0], velocity_max=lut.theta_dot[-1],
                torque_min=lut.tau_min, torque_max=lut.tau_max, **kwargs)

    @staticmethod
    def _cell(x, x_min, step, n):
        # index of the grid cell (0 ... n-2) and position inside it (0 ... 1)
        pos = (x - x_min) / step
        idx = ops.clip(ops.floor(pos), 0, n - 2)
        return ops.cast(idx, "int32"), pos - idx

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
        ia, fa = self._cell(angle, self.angle_min, self.angle_step, self.n_angles)
        iv, fv = self._cell(velocity, self.velocity_min, self.velocity_step, self.n_velocities)
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
            angle_min=self.angle_min, angle_max=self.angle_max,
            velocity_min=self.velocity_min, velocity_max=self.velocity_max,
            torque_min=self.torque_min.tolist(), torque_max=self.torque_max.tolist(),
        )
        return config