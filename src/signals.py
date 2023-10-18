import numpy as np
from scipy.sparse import rand


def compute_sparse(
    dim: tuple, values_range: tuple, density: float, seed: int = None
) -> np.ndarray:
    """
    Generate a sparse matrix with random values within the specified range.

    Args:
        dim (tuple): Dimensions (rows, columns) of the sparse matrix.
        values_range (tuple): Range (min, max) for the random values.
        density (float): Density of non-zero elements in the sparse matrix.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Sparse matrix with random values within the specified range.
    """
    value_min, value_max = np.min(values_range), np.max(values_range)
    spikes = rand(dim[0] - 2, dim[1] - 2, density, random_state=seed).toarray()
    spikes[spikes != 0] = spikes[spikes != 0] * (value_max - value_min) + value_min

    return np.pad(
        spikes,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )


def compute_smooth(
    dim: tuple,
    values_range: tuple,
    sigmas_range: tuple | list | float,
    nb_gaussian: int,
) -> np.ndarray:
    """
    Generate a smooth 2D array with Gaussian blobs.

    Args:
        dim (tuple): Dimensions (rows, columns) of the 2D array.
        values_range (tuple): Range (min, max) for the random values.
        sigmas_range (tuple | list | float): Range or value for standard deviations.
        nb_gaussian (int): Number of Gaussian blobs.

    Returns:
        np.ndarray: 2D array with Gaussian blobs.
    """
    if isinstance(sigmas_range, tuple):
        sigmas = np.random.uniform(*sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, list):
        sigmas = np.random.choice(sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, (float, int)):
        sigmas = sigmas * np.ones(nb_gaussian)
    else:
        ValueError("sigmas should be of type : tuple, list or int/float")

    amplitudes = np.random.uniform(size=nb_gaussian)
    centers = (1 - np.max(sigmas)) * np.random.uniform(-1, 1, (nb_gaussian, 2))

    x = np.linspace(-1, 1, dim[0])
    y = np.linspace(-1, 1, dim[1])
    x, y = np.meshgrid(x, y)
    grid_points = np.vstack((x.flatten(), y.flatten())).T

    smooth = np.zeros(dim)
    for s, c, a in zip(sigmas, centers, amplitudes):
        smooth += a * np.exp(
            -np.sum((grid_points - c) ** 2, axis=1) / (2 * s**2)
        ).reshape(dim)

    value_min, value_max = np.min(values_range), np.max(values_range)
    smooth = smooth / np.max(smooth) * (value_max - value_min) + value_min

    return smooth


def compute_y(y0: np.ndarray, psnr: int) -> np.ndarray:
    """
    Add noise to the input array to achieve a specified PSNR (Peak Signal-to-Noise Ratio).

    Args:
        y0 (np.ndarray): Input array.
        psnr (int): Target PSNR value.

    Returns:
        np.ndarray: Noisy version of the input array to achieve the specified PSNR.
    """
    y0_max = np.max(np.abs(y0))
    mse_db = 20 * np.log10(y0_max) - psnr
    mse = 10 ** (mse_db / 10)
    noise = np.random.normal(0, np.sqrt(mse / 2), y0.shape)
    return y0 + noise
