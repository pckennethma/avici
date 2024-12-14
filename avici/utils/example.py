from pathlib import Path
import numpy as onp
import logging
from avici.definitions import PROJECT_DIR
from avici.utils.parse import load_data_config
from avici.buffer import Sampler
from avici.utils.data import onp_standardize_data

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG to capture all logs
logger = logging.getLogger(__name__)

def simulate_data( d, n, *, n_interv=0, seed=0, domain=None, path=None, module_paths=None):
    """
    Helper function for simulating data from a pre-specified `domain` or a YAML domain configuration file at `path`.

    Args:
        d (int): number of variables in the system to be simulated
        n (int): number of observational data points to be sampled
        n_interv (int): number of interventional data points to be sampled
        seed (int): random seed
        domain (str): specifier of domain to be simulated. Currently implemented options:
            `lin-gauss`,
            `lin-gauss-heterosked`,
            `lin-laplace-cauchy`,
            `rff-gauss`,
            `rff-gauss-heterosked`,
            `rff-laplace-cauchy`,
            `gene-ecoli`
            `lin-gauss-ldp`
            (all `.yaml` file names inside `avici.config.examples`).
            Only one of `domain` and `path` must be specified.
        path (str): path to YAML domain configuration, like the examples in `avici.config`
        module_paths (str): path (or list of paths) to additional modules used in the domain configuration file

    Returns:
        tuple: the function returns a 3-tuple of
            - g (ndarray): `[d, d]` causal graph of `d` variables
            - x (ndarray): `[n + n_interv, d]` data matrix containing `n + n_interv` observations of the `d` variables
            - interv (ndarray): `[n + n_interv, d]` binary matrix indicating which nodes were intervened upon

    """
    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=seed))

    # load example domain specification
    if domain is not None:
        assert path is None, "Only specify one of `domain` and `path`"
        abspath = PROJECT_DIR / f"avici/config/examples/{domain}.yaml"
    elif path is not None:
        assert domain is None, "Only specify one of `domain` and `path`"
        abspath = Path(path)
    else:
        raise KeyError("Specify either an an `avici.config.examples` domain (YAML name) or a path to a YAML config.")

    if abspath.is_file():
        kwargs = dict(n_observations_obs=n, n_observations_int=n_interv)
        spec_tree = load_data_config(abspath, force_kwargs=kwargs, abspath=True,
                                     module_paths=module_paths, load_modules=True)["data"]
        spec = spec_tree[next(iter(spec_tree))]
    else:
        raise KeyError(f"`{abspath}` does not exist.")

    # sample and concatenate all data
    data = Sampler.generate_data(
        rng,
        n_vars=d,
        spec_list=spec,
    )
    x = onp.concatenate([data["x_obs"], data["x_int"]], axis=-3)

    # standardize only if not real-valued data
    data = onp_standardize_data(data) if not data["is_count_data"] else data

    if n_interv:
        return data["g"].astype(int), x[..., 0], x[..., 1]
    else:
        return data["g"].astype(int), x[..., 0], None

def simulate_ldp_data(d, n, *, n_interv=0, seed=0, domain=None, path=None, module_paths=None, epsilon=1.0, delta=1e-5):
    """
    Simulate data and add independent Gaussian noise to each variable for LDP.
    """
    # Call the existing simulate_data function
    g, x_obs, x_int = simulate_data(
        d, n, n_interv=n_interv, seed=seed, domain=domain, path=path, module_paths=module_paths
    )

    # Initialize RNG
    rng = onp.random.default_rng(onp.random.SeedSequence(entropy=seed))

    # Compute the noise scale based on epsilon and delta under the Gaussian mechanism
    sensitivity = 1  # Assuming L2 sensitivity is 1
    x_obs = onp.clip(x_obs, -0.5, 0.5)
    noise_scale = onp.sqrt(2 * onp.log(1.25 / delta)) * sensitivity / epsilon

    # Clip data into -0.5 to 0.5 to satisfy sensitivity=1 by using sigmoid(x) - 0.5
    x_obs += rng.normal(loc=0.0, scale=noise_scale, size=x_obs.shape)

    logger.info(f"d: {d}")
    logger.info(f"epsilon: {epsilon}")
    logger.info(f"delta: {delta}")
    logger.info(f"Noise scale: {noise_scale}")

    # Add Gaussian noise to observational data
    x_obs += rng.normal(loc=0.0, scale=noise_scale, size=x_obs.shape)

    # Return only the necessary values
    return g, x_obs, x_int