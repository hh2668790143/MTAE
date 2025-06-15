
import numpy as np
import pywt
from pykalman import KalmanFilter
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.signal import wiener
from scipy.sparse import diags

def Gamma_Noisy_return_N(x, snr):  # snr:信噪比
    # snr=350
    # print('Gamma')
    x_gamma = []
    x_gamma_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        # WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        # xpower = np.sum(signal ** 2 / len(signal))
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        npower = xpower / snr
        gamma = np.random.gamma(shape=2, size=len(signal)) * np.sqrt(npower)  # attention  shape=2
        # WavePlot_Single(gamma, 'gamma')

        x_gamma.append(x[i] + gamma)
        x_gamma_only.append(gamma)

    x_gamma = np.array(x_gamma)
    x_gamma_only = np.array(x_gamma_only)
    # x_gamma_only = np.expand_dims(x_gamma_only, 1)

    return x_gamma_only, x_gamma, x_gamma.shape[-1]


def Rayleign_Noisy_return_N(x, snr):  # snr:信噪比

    # snr=350
    x_rayleign = []
    x_rayleign_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        # xpower = np.sum(signal ** 2 / len(signal))
        # xpower = np.sum(signal ** 2 / len(signal))
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        npower = xpower / snr
        rayleign = np.random.rayleigh(size=len(signal)) * np.sqrt(npower)
        # WavePlot_Single(rayleign, 'rayleigh')

        x_rayleign.append(x[i] + rayleign)
        x_rayleign_only.append(rayleign)

    x_rayleign = np.array(x_rayleign)
    x_rayleign_only = np.array(x_rayleign_only)
    # x_rayleign_only = np.expand_dims(x_rayleign_only, 1)

    return x_rayleign_only, x_rayleign, x_rayleign.shape[-1]


def Exponential_Noisy_return_N(x, snr):  # snr:信噪比

    # snr=300
    x_exponential = []
    x_exponential_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        # xpower = np.sum(signal ** 2 / len(signal))
        # xpower = np.sum(signal ** 2 / len(signal))
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        npower = xpower / snr
        exponential = np.random.exponential(size=len(signal)) * np.sqrt(npower)
        # WavePlot_Single(exponential, 'exponential')

        x_exponential.append(x[i] + exponential)
        x_exponential_only.append(exponential)

    x_exponential = np.array(x_exponential)
    x_exponential_only = np.array(x_exponential_only)
    # x_exponential_only = np.expand_dims(x_exponential_only, 1)

    return x_exponential_only, x_exponential, x_exponential.shape[-1]


def Uniform_Noisy_return_N(x, snr):  # snr:信噪比

    # snr=250
    x_uniform = []
    x_uniform_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        # xpower = np.sum(signal ** 2 / len(signal))
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        # xpower = np.sum(signal ** 2 / len(signal))
        npower = xpower / snr
        uniform = np.random.uniform(size=len(signal)) * np.sqrt(npower)
        # WavePlot_Single(uniform, 'uniform')

        x_uniform.append(x[i] + uniform)
        x_uniform_only.append(uniform)

    x_uniform = np.array(x_uniform)
    x_uniform_only = np.array(x_uniform_only)
    # x_uniform_only = np.expand_dims(x_uniform_only, 1)

    return x_uniform_only, x_uniform, x_uniform.shape[-1]


def Poisson_Noisy_return_N(x, snr):  # snr:信噪比

    # print("possion")
    x_poisson = []
    x_poisson_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        # xpower = np.sum(signal ** 2 / len(signal))
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        # xpower = np.sum(signal ** 2 / len(signal))
        npower = xpower / snr
        poisson = np.random.poisson(1, len(signal)) * np.sqrt(npower)
        # WavePlot_Single(poisson, 'poisson')

        x_poisson.append(x[i] + poisson)
        x_poisson_only.append(poisson)

    x_poisson = np.array(x_poisson)
    x_poisson_only = np.array(x_poisson_only)
    # x_poisson_only = np.expand_dims(x_poisson_only, 1)

    return x_poisson_only, x_poisson, x_poisson.shape[-1]


def Gussian_Noisy_return_N(x, snr):  # snr:信噪比
    # snr=100
    x_gussian = []
    x_gussian_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(np.array([i ** 2 for i in signal]) / len(signal))
        npower = xpower / snr
        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        x_gussian.append(x[i] + gussian)
        x_gussian_only.append(gussian)

    x_gussian = np.array(x_gussian)
    x_gussian_only = np.array(x_gussian_only)

    return x_gussian_only, x_gussian, x_gussian.shape[-1]


def Kalman_1D(x):
    x_Kal = []
    # print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        # for j in range(x.shape[1]):
        #     signal = np.array(x[i][j])
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = Kalman1D(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    # x_Kal = np.expand_dims(x_Kal, 1)
    x_Kal_only = x - x_Kal

    return x_Kal_only, x_Kal, x_Kal.shape[-1]

def L1_1D(x):
    x_Kal = []
    # print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):

        signal = np.array(x[i])
        signal = np.squeeze(signal)

        signal_Kalman = l1_trend_filter(signal, lam=0.1)

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)

    x_Kal_only = x - x_Kal

    return x_Kal_only, x_Kal, x_Kal.shape[-1]

def Hp_1D(x):
    x_Kal = []
    # print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):

        signal = np.array(x[i])
        signal = np.squeeze(signal)

        signal_Kalman = hp_trend_filter(signal, lam=0.1)

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)

    x_Kal_only = x - x_Kal

    return x_Kal_only, x_Kal, x_Kal.shape[-1]

def Wiener_1D(x):
    x_Wie = []
    # print("WienerFiltering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        # WavePlot_Single(x[i],'signal')

        signal_Wie = wiener(signal, 81)

        # WavePlot_Single(signal_sav,'kalman')

        x_Wie.append(signal_Wie)

    x_Wie = np.array(x_Wie)
    # x_Wie = np.expand_dims(x_Wie, 1)
    x_Wie_only = x - x_Wie

    return x_Wie_only, x_Wie, x_Wie.shape[-1]

def Kalman1D(observations, damping=1):  # [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.01
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    pred_state = np.squeeze(pred_state)
    return pred_state


def l1_trend_filter(signal, lam):
    n = len(signal)
    D = toeplitz(np.concatenate(([1], np.zeros(n - 1))), signal)
    w = np.ones(n)

    def objective(x):
        return 0.5 * np.sum((x - signal) ** 2) + lam * np.sum(w * np.abs(D @ x))

    def gradient(x):
        return x - signal + lam * D.T @ (w * np.sign(D @ x))

    result = minimize(objective, signal, jac=gradient, method='L-BFGS-B')
    filtered_signal = result.x

    return filtered_signal

def hp_trend_filter(signal, lam):
    n = len(signal)
    D = diags([-1, 2, -1], [-1, 0, 1], shape=(n - 2, n))
    I = np.identity(n)
    trend_filter = lam * (D.T @ D) + I
    signal = signal.astype(np.float32)
    trend = np.linalg.solve(trend_filter, signal)

    return trend

if __name__ == '__main__':
    t = np.linspace(0, 10, 100)
    signal = np.sin(t) + np.random.normal(0, 0.1, 100)
    signal = np.expand_dims(signal, axis=0)

    # 应用L1趋势滤波器
    # filtered_signal = L1_1D(signal)
    filtered_signal = Hp_1D(signal)
    # filtered_signal = Kalman_1D(signal)

    print(1)