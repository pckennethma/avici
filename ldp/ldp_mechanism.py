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
    # Compute robust statistics
    med = onp.median(x)
    mad = onp.median(onp.abs(x - med)) + 1e-10  # add small term for numerical stability

    # Determine linear scaling bounds
    lower_bound = med - c * mad
    upper_bound = med + c * mad

    # Scale to [-0.5, 0.5]
    scaled = (x - lower_bound) / (upper_bound - lower_bound) - 0.5
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
        x_obs = original_data.x_obs

        sensitivity: float
        if self.clipping_scheme == ClippingScheme.CONSTANT:
            x_obs = onp.clip(x_obs, -0.5, 0.5)
            sensitivity = 1
        elif self.clipping_scheme == ClippingScheme.SIGMOID:
            x_obs = sigmoid(x_obs) - 0.5
            sensitivity = 1
        else:
            raise ValueError(f"Invalid clipping scheme: {self.clipping_scheme}")

        if isinstance(self.delta, str) and self.delta == "inverse":
            self.delta = 1 / (n_observations_obs + n_observations_int)

        noise_scale = onp.sqrt(2 * onp.log(1.25 / self.delta)) * sensitivity / self.epsilon
        x_obs += rng.normal(loc=0, scale=noise_scale, size=x_obs.shape)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )

