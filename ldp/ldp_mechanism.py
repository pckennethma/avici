import numpy as onp
from enum import Enum
from avici.synthetic import GraphModel, MechanismModel
from avici.synthetic import Data
from ldp.xiao import hybrid_mechanism

def sigmoid(x):
    x = onp.clip(x, -500, 500)  # Clamp input range
    return onp.where(x >= 0,
        1 / (1 + onp.exp(-x)),
        onp.exp(x) / (1 + onp.exp(x)))

def robust_linear_clip(x, c=3.0, constant=0.5):
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

    # Scale each column to [-range_bound, range_bound]
    scaled = ((x - lower_bound) / range_bound) * 2 * constant - constant
    # Clip values outside [-constant, constant]
    scaled = onp.clip(scaled, -constant, constant)
    return scaled, lower_bound, upper_bound

def rescale(scaled, lower_bound, upper_bound, constant=0.5):
    range_bound = upper_bound - lower_bound
    return (scaled + constant) / (2 * constant) * range_bound + lower_bound

class ClippingScheme(str, Enum):
    CONSTANT = "constant" # clip to [-0.5, 0.5]
    SIGMOID = "sigmoid" # clip to [-0.5, 0.5] using sigmoid(x) - 0.5
    ROBUST_LINEAR_CLIP = "robust_linear_clip" # clip to [-0.5, 0.5] using robust linear clipping

class GaussianLDP(MechanismModel):
    def __init__(
            self, 
            epsilon: float | list[float], 
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

        if isinstance(self.epsilon, list):
            eps = onp.random.choice(self.epsilon)
        else:
            eps = self.epsilon

        noise_scale = onp.sqrt(2 * onp.log(1.25 / self.delta)) * sensitivity / eps
        x_obs_val += rng.normal(loc=0, scale=noise_scale, size=x_obs_val.shape)

        # merge x_obs_val with the intervention indicator in original_data.x_obs
        x_obs = onp.stack([x_obs_val, original_data.x_obs[:,:,1]], axis=-1)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )

class LaplaceLDP(MechanismModel):
    def __init__(
            self, 
            epsilon: float | list[float], 
            base_mechanism: MechanismModel,
    ):
        self.epsilon = epsilon
        self.base_mechanism = base_mechanism

    def __call__(self, rng: onp.random.Generator, g: onp.ndarray, n_observations_obs : int, n_observations_int: int):

        epsilon = onp.random.choice(self.epsilon) if isinstance(self.epsilon, list) else self.epsilon

        original_data = self.base_mechanism(rng, g, n_observations_obs, n_observations_int)
        x_obs_val = original_data.x_obs[:,:,0]

        x_clipped, lower_bound, upper_bound = robust_linear_clip(x_obs_val, constant=0.5)
        noise_scale = 1 / epsilon
        noise = rng.laplace(loc=0, scale=noise_scale, size=x_clipped.shape)
        x_perturbed = x_clipped + noise
        x_rescaled = rescale(x_perturbed, lower_bound, upper_bound, constant=0.5)

        # merge x_obs_val with the intervention indicator in original_data.x_obs
        x_obs = onp.stack([x_rescaled, original_data.x_obs[:,:,1]], axis=-1)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )


class HMLDP(MechanismModel):
    def __init__(
            self, 
            epsilon: float | list[float], 
            base_mechanism: MechanismModel,
    ):
        self.epsilon = epsilon
        self.base_mechanism = base_mechanism

    def __call__(self, rng: onp.random.Generator, g: onp.ndarray, n_observations_obs : int, n_observations_int: int):

        epsilon = onp.random.choice(self.epsilon) if isinstance(self.epsilon, list) else self.epsilon

        original_data = self.base_mechanism(rng, g, n_observations_obs, n_observations_int)
        x_obs_val = original_data.x_obs[:,:,0]

        x_clipped, lower_bound, upper_bound = robust_linear_clip(x_obs_val, constant=1)
        x_perturbed = hybrid_mechanism(x_clipped, epsilon, rng=rng)
        x_rescaled = rescale(x_perturbed, lower_bound, upper_bound, constant=1)

        # merge x_obs_val with the intervention indicator in original_data.x_obs
        x_obs = onp.stack([x_rescaled, original_data.x_obs[:,:,1]], axis=-1)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )