import numpy as onp
from enum import Enum
from avici.synthetic import GraphModel, MechanismModel
from avici.synthetic import Data

def sigmoid(x):
    x = onp.clip(x, -500, 500)  # Clamp input range
    return onp.where(x >= 0,
        1 / (1 + onp.exp(-x)),
        onp.exp(x) / (1 + onp.exp(x)))

def robust_linear_clip(x, c=3.0):
    # x is a (n, d) array
    # Compute median per column (variable)
    med = onp.median(x, axis=0, keepdims=True)  # shape: (1, d)

    abs_devs = onp.abs(x - med)
    # Compute MAD per column (variable)
    mad = onp.median(abs_devs, axis=0, keepdims=True) + 1e-10  # shape: (1, d)

    # Determine linear scaling bounds per column
    lower_bound = med - c * mad  # shape: (1, d)
    upper_bound = med + c * mad  # shape: (1, d)
    range_bound = upper_bound - lower_bound  # shape: (1, d)

    # Scale each column to [-0.5, 0.5]
    scaled = ((x - lower_bound) / range_bound) - 0.5

    # Clip values outside [-0.5, 0.5]
    scaled = onp.clip(scaled, -0.5, 0.5)
    return scaled

class ClippingScheme(str, Enum):
    CONSTANT = "constant" # clip to [-0.5, 0.5]
    SIGMOID = "sigmoid" # clip to [-0.5, 0.5] using sigmoid(x) - 0.5
    ROBUST_LINEAR_CLIP = "robust_linear_clip" # clip to [-0.5, 0.5] using robust linear clipping

class GaussianLDP(MechanismModel):
    def __init__(
            self, 
            epsilon: float, 
            delta: float | str, 
            clipping_scheme: ClippingScheme,
            base_mechanism: MechanismModel,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.base_mechanism = base_mechanism
        self.clipping_scheme = clipping_scheme

    def __call__(self, rng: onp.random.Generator, g: onp.ndarray, n_observations_obs : int, n_observations_int: int):
        
        original_data = self.base_mechanism(rng, g, n_observations_obs, n_observations_int)
        x_obs_val = original_data.x_obs[:,:,0]

        sensitivity: float
        if self.clipping_scheme == ClippingScheme.CONSTANT:
            x_obs_val = onp.clip(x_obs_val, -0.5, 0.5)
            sensitivity = 1
        elif self.clipping_scheme == ClippingScheme.SIGMOID:
            x_obs_val = sigmoid(x_obs_val) - 0.5
            sensitivity = 1
        elif self.clipping_scheme == ClippingScheme.ROBUST_LINEAR_CLIP:
            x_obs_val = robust_linear_clip(x_obs_val)
            sensitivity = 1
        else:
            raise ValueError(f"Invalid clipping scheme: {self.clipping_scheme}")

        if isinstance(self.delta, str) and self.delta == "inverse":
            self.delta = 1 / (n_observations_obs + n_observations_int)

        noise_scale = onp.sqrt(2 * onp.log(1.25 / self.delta)) * sensitivity / self.epsilon
        x_obs_val += rng.normal(loc=0, scale=noise_scale, size=x_obs_val.shape)

        # merge x_obs_val with the intervention indicator in original_data.x_obs
        x_obs = onp.stack([x_obs_val, original_data.x_obs[:,:,1]], axis=-1)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )

