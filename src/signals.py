import numpy as np
from scipy.sparse import rand

__all__ = ["compute_sparse", "compute_smooth", "compute_y"]

def compute_sparse(
    N: int, values_range: tuple, density: float, seed: int = None
) -> np.ndarray:
    """
    Generate a sparse matrix with random values within the specified range.

    Args:
        N (int): Size of the squared sparse matrix.
        values_range (tuple): Range (min, max) for the random values.
        density (float): Density of non-zero elements in the sparse matrix.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Sparse matrix with random values within the specified range.
    """
    margin = int(0.05 * N)
    value_min, value_max = np.min(values_range), np.max(values_range)
    spikes = rand(N - 2 * margin, N - 2 * margin, density, random_state=seed).toarray()
    spikes[spikes != 0] = spikes[spikes != 0] * (value_max - value_min) + value_min

    return np.pad(
        spikes,
        ((margin, margin), (margin, margin)),
        mode="constant",
        constant_values=0,
    )


def compute_smooth(
    N: int,
    smooth_amplitude: float,
    sigmas_range: tuple | list | float,
    nb_gaussian: int,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a smooth 2D array with Gaussian blobs.

    Args:
        N (tuple): Size of the squared 2D array.
        values_range (tuple): Range (min, max) for the random values.
        sigmas_range (tuple | list | float): Range or value for standard deviations.
        nb_gaussian (int): Number of Gaussian blobs.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: 2D array with Gaussian blobs.
    """
    np.random.seed(seed)
    if isinstance(sigmas_range, tuple):
        sigmas = np.random.uniform(*sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, list):
        sigmas = np.random.choice(sigmas_range, nb_gaussian)
    elif isinstance(sigmas_range, (float, int)):
        sigmas = sigmas * np.ones(nb_gaussian)
    else:
        ValueError("sigmas should be of type : tuple, list or int/float")

    amplitudes = np.random.uniform(-1, 1, size=nb_gaussian)
    centers = (1 - 1.5 * np.max(sigmas)) * np.random.uniform(-1, 1, (nb_gaussian, 2))

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    x, y = np.meshgrid(x, y)
    grid_points = np.vstack((x.flatten(), y.flatten())).T

    smooth = np.zeros((N, N))
    for s, c, a in zip(sigmas, centers, amplitudes):
        smooth += a * np.exp(
            -np.sum((grid_points - c) ** 2, axis=1) / (2 * s**2)
        ).reshape((N, N))

    smooth = smooth_amplitude * smooth / np.max(np.abs(smooth))

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
    # y0_max = np.max(np.abs(y0))
    # mse_db = 20 * np.log10(y0_max) - psnr
    # mse = 10 ** (mse_db / 10)
    # noise = np.random.normal(0, np.sqrt(mse / 2), y0.shape)
    import pyxu.util.complex as pxuc
    y = pxuc.view_as_complex(y0)
    sigma = np.abs(y).max()**2 * (10 ** (-psnr / 10)) / y.size
    noise = np.random.normal(0, np.sqrt(sigma/2), y0.shape)
    return y0 + noise
