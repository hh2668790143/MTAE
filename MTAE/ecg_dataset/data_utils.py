import numpy as np
import scipy.interpolate as spi
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def down_sample(signal, sample_rate_1, sample_rate_2):  # sample_rate_1 ----> sample_rate_2
    X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_1))  # 4s  步长0.004对应250HZ  1000个点
    new_X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_2))  # 4S  步长0.002对应500HZ  2000个点
    ipo3 = spi.splrep(X[:len(signal)], signal, k=3)
    iy3 = spi.splev(new_X, ipo3)

    return iy3

def time_warp(x, sigma=0.03, knot=10, smooth=True, smooth_sigma=0.5):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, 1))

    random_warps = np.tile(random_warps, (1, 1, x.shape[2]))

    if smooth:
        random_warps = gaussian_filter1d(random_warps, smooth_sigma, axis=1)

    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret

def preprocess_signals(x_train, method='minmax'):
    # 根据选择的缩放方法来标准化数据
    if method == 'minmax':
        scaler = MinMaxScaler()  # 将数据缩放到 [0, 1] 范围
    elif method == 'standard':
        scaler = StandardScaler()  # 将数据标准化为均值 0，方差 1
    else:
        raise ValueError("Unsupported method. Use 'minmax' or 'standard'.")

    # 将每个样本进行缩放
    batch_size, signal_length, num_dimensions = x_train.shape
    X_flat = x_train.reshape(batch_size * signal_length, num_dimensions)  # 扁平化数据以适应Scaler
    X_scaled = scaler.fit_transform(X_flat)  # 进行缩放
    X_scaled = X_scaled.reshape(batch_size, signal_length, num_dimensions)  # 恢复原始形状
    return X_scaled
