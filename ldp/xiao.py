import numpy as np
from numpy.random import Generator
from typing import Union, List, Optional, Literal

def duchi_one_dimension(t: float, epsilon: float, rng: Generator) -> float:
    """
    Duchi et al.'s method for one-dimensional numeric data.

    Parameters:
    - t: float in [-1, 1]
    - epsilon: float > 0
    - rng: numpy.random.Generator instance for randomness

    Returns:
    - t_star: float in { -X, X } where X = (e^epsilon + 1)/(e^epsilon - 1)
    """
    exp_eps = np.exp(epsilon)
    exp_eps_plus1 = exp_eps + 1
    exp_eps_minus1 = exp_eps - 1
    X = exp_eps_plus1 / exp_eps_minus1
    p = (t * exp_eps_minus1 + exp_eps_plus1) / (2 * exp_eps_plus1)
    return X if rng.random() < p else -X

def pm_one_dimension(t: float, epsilon: float, rng: Generator) -> float:
    """
    Piecewise Mechanism for one-dimensional numeric data (PM).

    Parameters:
    - t: float in [-1, 1]
    - epsilon: float > 0
    - rng: numpy.random.Generator instance for randomness

    Returns:
    - t_star: float in [-C, C]

    Where C = (e^(epsilon/2)+1)/(e^(epsilon/2)-1).
    """
    exp_half_eps = np.exp(epsilon / 2)
    C = (exp_half_eps + 1) / (exp_half_eps - 1)
    ell = ((C + 1) / 2) * t - (C - 1) / 2
    r = ell + (C - 1)
    center_prob = exp_half_eps / (exp_half_eps + 1)

    if rng.random() < center_prob:
        return rng.uniform(ell, r)
    else:
        length_left = ell + C
        length_right = C - r
        total_length = length_left + length_right
        u = rng.uniform(0, total_length)
        if u < length_left:
            return -C + u
        else:
            return r + (u - length_left)

def var_pm_max(epsilon: float) -> float:
    """
    Compute the maximum variance for PM as a reference.

    Parameters:
    - epsilon: float > 0

    Returns:
    - Maximum variance: float
    """
    exp_half_eps = np.exp(epsilon / 2)
    denom = (exp_half_eps - 1) ** 2
    return (4 * exp_half_eps) / (3 * denom)

def hybrid_one_dimension(t: float, epsilon: float, rng: Generator) -> float:
    """
    Hybrid Mechanism (HM) for one-dimensional numeric data.

    Parameters:
    - t: float in [-1, 1]
    - epsilon: float > 0
    - rng: numpy.random.Generator instance for randomness

    Returns:
    - t_star: float perturbed under epsilon-LDP
    """
    epsilon_star = 0.61

    if epsilon <= epsilon_star:
        return duchi_one_dimension(t, epsilon, rng)
    else:
        alpha = 1 - np.exp(-epsilon / 2)
        if rng.random() < alpha:
            return pm_one_dimension(t, epsilon, rng)
        else:
            return duchi_one_dimension(t, epsilon, rng)

def hybrid_mechanism_tuple(
    t: np.ndarray,
    epsilon: float,
    rng: Generator,
    method: Literal['PM', 'HM'] = 'PM',
    k_choice: Literal['full', 'auto'] = 'full'
) -> np.ndarray:
    """
    Our Method for Multiple Numeric Attributes.

    Parameters:
    - t: numpy array of shape (d,) with each t[i] in [-1, 1]
    - epsilon: float > 0
    - rng: numpy.random.Generator instance for randomness
    - method: 'PM' or 'HM'
    - k_choice: 'full' or 'auto'

    Returns:
    - t_star: numpy array of shape (d,) perturbed under epsilon-LDP
    """
    d = t.size
    k = d if k_choice == 'full' else max(1, min(d, int(np.floor(epsilon / 2.5))))
    chosen = rng.choice(d, k, replace=False)
    t_star = np.zeros(d)
    epsilon_per_k = epsilon / k
    scaling_factor = d / k

    if method == 'PM':
        perturb_fn = lambda val: pm_one_dimension(val, epsilon_per_k, rng)
    elif method == 'HM':
        perturb_fn = lambda val: hybrid_one_dimension(val, epsilon_per_k, rng)
    else:
        raise ValueError("Unknown method")

    for j in chosen:
        t_star[j] = scaling_factor * perturb_fn(t[j])
    return t_star

def hybrid_mechanism(
    t: np.ndarray,
    epsilon: float,
    rng: Generator,
    method: Literal['PM', 'HM'] = 'PM',
    k_choice: Literal['full', 'auto'] = 'full'
) -> np.ndarray:
    """
    Our Method for Multiple Numeric Attributes applied to a batch.

    Parameters:
    - t: numpy array of shape (n, d) with each t[i, j] in [-1, 1]
    - epsilon: float > 0
    - rng: numpy.random.Generator instance for randomness
    - method: 'PM' or 'HM'
    - k_choice: 'full' or 'auto'

    Returns:
    - t_star: numpy array of shape (n, d) perturbed under epsilon-LDP
    """
    n, _ = t.shape
    return np.array([hybrid_mechanism_tuple(ti, epsilon, rng, method, k_choice) for ti in t])

# Example usage:
if __name__ == "__main__":
    # Initialize a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    t = np.array([0.2, -0.5, 0.9, 0.1, -0.2])
    epsilon = 100

    # Use HM
    t_star_hm = hybrid_mechanism(t, epsilon, rng, method='HM')
    print("t_star with HM:", t_star_hm)

    # Use PM
    t_star_pm = hybrid_mechanism(t, epsilon, rng, method='PM')
    print("t_star with PM:", t_star_pm)