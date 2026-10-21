import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="lut_nn")
class LUT_preprocessing(keras.layers.Layer):
    """Map physical inputs (W, angle [deg], torque [Nm]) to the features of the network.

    Outputs:
      features        (batch, 3): [W, angle_norm, torque_norm], all in [0, 1]
      torque_norm_raw (batch, 1): torque_norm before the clip (<0 or >1 = infeasible request)
    """

    def __init__(self, angle_min, angle_max, torque_min, torque_max, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.torque_min = [float(v) for v in torque_min]
        self.torque_max = [float(v) for v in torque_max]

        self.n_angles = len(self.torque_min)
        self.angle_step = (self.angle_max - self.angle_min) / (self.n_angles - 1)

    @classmethod
    def from_lut(cls, lut, **kwargs):
        if lut.angle is None:
            raise RuntimeError("LUT not loaded: call lut.load_data() first")

        angles = np.unique(lut.angle)       # 37 grid angles, sorted
        steps = np.diff(angles)             # 36 steps between consecutive angles
        
        # control shape is correct -> FORSE SI POTREBBE RIMUOVERE
        if not np.allclose(steps, steps[0]):
            raise ValueError("The angle grid of the LUT is not uniform")
        if len(angles) != len(lut.torque_min) or len(angles) != len(lut.torque_max):
            raise ValueError("torque_min / torque_max must have one value per grid angle")

        return cls(angle_min=angles[0], angle_max=angles[-1],
                   torque_min=lut.torque_min, torque_max=lut.torque_max, **kwargs)

    def _interp(self, table, idx, frac):
        # linear interpolation of a 1D table at fractional index idx + frac
        t = ops.convert_to_tensor(np.asarray(table, dtype="float32"))
        return ops.take(t, idx) * (1.0 - frac) + ops.take(t, idx + 1) * frac

    def call(self, inputs):
        W, angle, torque = inputs  # each (batch, 1)

        angle = ops.clip(angle, self.angle_min, self.angle_max)
        pos = (angle - self.angle_min) / self.angle_step
        idx = ops.clip(ops.floor(pos), 0, self.n_angles - 2)
        frac = pos - idx
        idx = ops.cast(idx, "int32")

        torque_min = self._interp(self.torque_min, idx, frac)
        torque_max = self._interp(self.torque_max, idx, frac)

        torque_norm_raw = (torque - torque_min) / (torque_max - torque_min)
        torque_norm = ops.clip(torque_norm_raw, 0.0, 1.0)
        angle_norm = (angle - self.angle_min) / (self.angle_max - self.angle_min)
        W = ops.clip(W, 0.0, 1.0)

        features = ops.concatenate([W, angle_norm, torque_norm], axis=-1)
        return features, torque_norm_raw

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        return (batch, 3), (batch, 1)

    def get_config(self):
        config = super().get_config()      # name, dtype, trainable, ...
        config.pop("trainable", None)      # __init__ already sets trainable=False
        config.update(
            angle_min=self.angle_min, angle_max=self.angle_max,
            torque_min=self.torque_min, torque_max=self.torque_max,
        )
        return config