import numpy as onp
from enum import Enum
from avici.synthetic import GraphModel, MechanismModel
from avici.synthetic import Data


class ClippingScheme(str, Enum):
    CONSTANT = "constant" # clip to [-0.5, 0.5]
    SIGMOID = "sigmoid" # clip to [-0.5, 0.5] using sigmoid(x) - 0.5

class GaussianLDP(MechanismModel):
    def __init__(
            self, 
            epsilon: float, 
            delta: float, 
            clipping_scheme: ClippingScheme,
            base_mechanism: MechanismModel,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.base_mechanism = base_mechanism
        self.clipping_scheme = clipping_scheme

    def __call__(self, rng: onp.random.Generator, g, n_observations_obs, n_observations_int):
        
        original_data = self.base_mechanism(rng, g, n_observations_obs, n_observations_int)
        x_obs = original_data.x_obs

        sensitivity: float
        if self.clipping_scheme == ClippingScheme.CONSTANT:
            x_obs = onp.clip(x_obs, -0.5, 0.5)
            sensitivity = 1
        elif self.clipping_scheme == ClippingScheme.SIGMOID:
            x_obs = onp.clip(onp.sigmoid(x_obs) - 0.5, -0.5, 0.5)
            sensitivity = 1
        else:
            raise ValueError(f"Invalid clipping scheme: {self.clipping_scheme}")

        noise_scale = onp.sqrt(2 * onp.log(1.25 / self.delta)) * sensitivity / self.epsilon
        x_obs += rng.normal(loc=0, scale=noise_scale, size=x_obs.shape)

        return Data(
            x_obs=x_obs,
            x_int=original_data.x_int,
            is_count_data=original_data.is_count_data,
        )

