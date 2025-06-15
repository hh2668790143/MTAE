
import scipy.interpolate as spi
import numpy as np
from scipy.spatial.distance import pdist, squareform


def down_sample(signal, sample_rate_1, sample_rate_2):  # sample_rate_1 ----> sample_rate_2
    X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_1))  # 4s  步长0.004对应250HZ  1000个点
    new_X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_2))  # 4S  步长0.002对应500HZ  2000个点
    ipo3 = spi.splrep(X[:len(signal)], signal, k=3)
    iy3 = spi.splev(new_X, ipo3)

    return iy3

def time_warp(x, sigma=0.1, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def gaussian_kernel(x, y, sigma):
    """Compute the Gaussian RBF kernel between x and y."""
    return np.exp(-np.sum((x - y) ** 2, axis=1) / (2 * (sigma ** 2)))

def mmd_rbf(samples1, samples2, sigma):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples
    using the RBF kernel.

    :param samples1: Numpy array of samples from the first distribution (n x d)
    :param samples2: Numpy array of samples from the second distribution (m x d)
    :param sigma: Kernel width parameter
    """
    n, m = samples1.shape[0], samples2.shape[0]

    # Compute the kernel matrices for each distribution
    K_xx = gaussian_kernel(samples1, samples1, sigma)
    K_yy = gaussian_kernel(samples2, samples2, sigma)
    K_xy = gaussian_kernel(samples1, samples2, sigma)

    # Calculate the MMD^2 using the formula provided
    MMD2 = (np.mean(K_xx) - np.mean(K_xy) ** 2) / (n * (n - 1)) + \
           (np.mean(K_yy) - np.mean(K_xy) ** 2) / (m * (m - 1))

    return MMD2

if __name__ == '__main__':
    np.random.seed(42)
    samples1 = np.random.normal(size=(100, 2))
    samples2 = np.random.normal(size=(100, 2))

    # Define the kernel width (sigma)
    sigma = np.sqrt(2) * np.std(samples1, axis=0).mean()

    # Calculate MMD
    mmd_value = mmd_rbf(samples1, samples2, sigma)
    print(f"MMD^2: {mmd_value}")