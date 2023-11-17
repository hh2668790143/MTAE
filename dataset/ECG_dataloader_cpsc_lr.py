import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data

from dataset.load_data_cpsc2021 import load_data_cpsc2021, load_CPSC2021_data

from dataset.getHRV import get_nni_24H_1
from dataset.transformer import r_plot, paa, rescale

from sklearn.model_selection import train_test_split
import pywt
# import librosa
import scipy
# import yaml
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter, wiener

from pykalman import KalmanFilter
import math

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
# from PHM import get_data_num, preprocessing


# *****************ECG Func   start*****************


def getFloderK(data, folder, label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label == 0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y = np.zeros((remain_data.shape[0], 1))
    elif label == 1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data, folder_data_y, remain_data, remain_data_y


def getPercent(data_x, data_y, percent, seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=percent, random_state=seed)
    return train_x, test_x, train_y, test_y


def get_full_data(dataloader):
    full_data_x = []
    full_data_y = []
    for batch_data in dataloader:
        batch_x, batch_y = batch_data[0], batch_data[1]
        batch_x = batch_x.numpy()
        batch_y = batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i, 0, :])
            full_data_y.append(batch_y[i])

    full_data_x = np.array(full_data_x)
    full_data_y = np.array(full_data_y)
    assert full_data_x.shape[0] == full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x, full_data_y


def data_aug(train_x, train_y, times=2):
    res_train_x = []
    res_train_y = []
    for idx in range(train_x.shape[0]):
        x = train_x[idx]
        y = train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):
            x_aug = aug_ts(x)
            res_train_x.append(x_aug)
            res_train_y.append(y)

    res_train_x = np.array(res_train_x)
    res_train_y = np.array(res_train_y)

    return res_train_x, res_train_y


def aug_ts(x):
    left_ticks_index = np.arange(0, 140)
    right_ticks_index = np.arange(140, 319)
    np.random.shuffle(left_ticks_index)
    np.random.shuffle(right_ticks_index)
    left_up_ticks = left_ticks_index[:7]
    right_up_ticks = right_ticks_index[:7]
    left_down_ticks = left_ticks_index[7:14]
    right_down_ticks = right_ticks_index[7:14]

    x_1 = np.zeros_like(x)
    j = 0
    for i in range(x.shape[1]):
        if i in left_down_ticks or i in right_down_ticks:
            continue
        elif i in left_up_ticks or i in right_up_ticks:
            x_1[:, j] = x[:, i]
            j += 1
            x_1[:, j] = (x[:, i] + x[:, i + 1]) / 2
            j += 1
        else:
            x_1[:, j] = x[:, i]
            j += 1
    return x_1


# *****************ECG Func  end*****************

# **********************MEL strat ******************************
# params = yaml.load(open('./Mel_Para.yaml'))
extract_params = {
    'audio_len_s': 2,
    'eps': 2.220446049250313e-16,
    'fmax': 22050,
    'fmin': 0,
    'fs': 44100,
    'hop_length_samples': 82,  # 影响第三维
    'load_mode': 'varup',
    'log': True,
    'mono': True,
    'n_fft': 1024,
    'n_mels': 96,
    'normalize_audio': True,
    'patch_hop': 50,
    'patch_len': 100,
    'spectrogram_type': True,
    'win_length_samples': 564,
    'spectrogram_type': 'power',
    'audio_len_samples': 88200
}


def wavelet_preprocessing_set(X, waveletLevel=1, waveletFilter='haar'):  ##(1221,26,512)
    '''
    :param X: (sample_num, feature_num, sequence_length)
    :param waveletLevel:
    :param waveletFilter:
    :return: result (sample_num, extended_sequence_length, feature_num)
    '''

    if len(X.shape) == 2:
        X = np.expand_dims(X, 1)  # (292,1,140)

    N = X.shape[0]  # 1121
    feature_dim = X.shape[1]  # 26
    length = X.shape[2]  # 512

    signal_length = []
    stats = []
    extened_X = []
    # extened_X.append(X)  #（1221，512，26）

    for i in range(N):  # for each sample
        for j in range(feature_dim):  # for each dim
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter,
                                        level=waveletLevel)  # X(1221,26,512)    512-->(32,32,64,128,256)
            # 多尺度一维小波分解。  返回信号X在N层的小波分解
            if i == 0 and j == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])  # 256 128 64 32 32
                    signal_length.append(current_length)
                    extened_X.append(np.zeros((N, feature_dim, current_length)))
            for l in range(waveletLevel):
                extened_X[l][i, j, :] = wavelet_list[waveletLevel - l]  # (1221,512,26)-->(1221,256,26) .....(,32,)

    result = None
    first = True
    for mat in extened_X:
        mat_mean = mat.mean()
        mat_std = mat.std()
        mat = (mat - mat_mean) / mat_std
        stats.append((mat_mean, mat_std))
        if first:
            result = mat
            first = False
        else:
            result = np.concatenate((result, mat), -1)  # 512+256+128+64+32=992

        # result_f = np.flipud(result)
        res = np.concatenate((result, result), -1)

    # return result, signal_length
    return res, signal_length


#
# def mel_spectrogram_precessing_set(audio, params_extract=None):
#     """
#
#     :param audio:
#     :param params_extract:
#     :return:
#     """
#     audio=np.array(audio)
#     # make sure rows are channels and columns the samples
#     Mel_Matrix=[]
#     for idx in range(audio.shape[0]):
#
#         audio_idx = audio[idx].reshape([1, -1])
#         window = scipy.signal.hamming(params_extract.get('win_length_samples'), sym=False)  #window len 1764
#
#         mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),     #梅尔滤波器 将能量谱转换为梅尔频率
#                                         n_fft=params_extract.get('n_fft'),
#                                         n_mels=params_extract.get('n_mels'),
#                                         fmin=params_extract.get('fmin'),
#                                         fmax=params_extract.get('fmax'),
#                                         htk=False,
#                                         norm=None)
#
#         # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
#         feature_matrix = np.empty((0, params_extract.get('n_mels')))
#         for channel in range(0, audio_idx.shape[0]):
#             spectrogram = get_spectrogram(       #梅尔谱图
#                 y=audio_idx[channel, :],
#                 n_fft=params_extract.get('n_fft'),
#                 win_length_samples=params_extract.get('win_length_samples'),
#                 hop_length_samples=params_extract.get('hop_length_samples'),
#                 spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
#                 center=True,
#                 window=window,
#                 params_extract=params_extract
#             )
#
#             mel_spectrogram = np.dot(mel_basis, spectrogram)   #梅尔频率点乘梅尔谱图=梅尔频谱图
#             mel_spectrogram = mel_spectrogram.T
#
#             if params_extract.get('log'):
#                 mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))
#
#             feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)
#
#             Mel_Matrix.append(feature_matrix)
#     Mel_Matrix=np.expand_dims(Mel_Matrix,1)
#
#     #print()
#
#     return Mel_Matrix,Mel_Matrix.shape[1:]
#
#
# def get_spectrogram(y,
#                     n_fft=1024,
#                     win_length_samples=0.04,
#                     hop_length_samples=0.02,
#                     window=scipy.signal.hamming(512, sym=False),
#                     center=True,
#                     spectrogram_type='magnitude',
#                     params_extract=None):
#
#     if spectrogram_type == 'power':
#         return np.abs(librosa.stft(y + params_extract.get('eps'),        #STFT  短时傅里叶变换  时频信号转换
#                                    n_fft=n_fft,
#                                    win_length=win_length_samples,
#                                    hop_length=hop_length_samples,
#                                    center=center,
#                                    window=window)) ** 2

# **********************MEL end******************************

def RP_preprocessing_set(X_train):
    ##################
    # Down-sample
    ##################
    signal_dim = X_train.shape[-1]
    if signal_dim > 500:
        down_scale = X_train.shape[-1] // 128
    else:
        down_scale = X_train.shape[-1] // 32

    (size_H, size_W) = (X_train.shape[-1] // down_scale, X_train.shape[-1] // down_scale)
    print('[INFO] Raw Size: {}'.format((X_train.shape[-1], X_train.shape[-1])))
    print('[INFO] Downsample Size: {}'.format((size_H, size_W)))

    X_train_ds = paa(X_train, down_scale)

    # 2.1.RP image
    X_train_rp = np.empty(shape=(len(X_train_ds), size_H, size_W), dtype=np.float32)

    for i in range(len(X_train_ds)):
        X_train_rp[i, :, :] = r_plot(X_train_ds[i, :])

    X_train_rp = np.expand_dims(X_train_rp, 1)
    return X_train_rp, X_train_rp.shape[1:]


import time


class EM_FK():

    def __init__(self, A, C, Q, R, B, D, m0, P0, random_state):
        self.A = A  # transition_matrix
        self.C = C  # observation_matrix
        self.Q = Q  # transition_covariance
        self.R = R  # observation_covariance
        self.B = B  # transition_offset
        self.D = D  # observation_offset
        self.m = m0  # initial_state_mean
        self.p = P0  # initial_state_covariance
        self.random_state = random_state

    def filter(self, x):
        kf = KalmanFilter(self.A, self.C, self.Q, self.R, self.B, self.D, self.m, self.p, self.random_state)

        filtered_state_estimater, nf_cov = kf.filter(x)
        smoothed_state_estimater, ns_cov = kf.smooth(x)

        pred_state = np.squeeze(smoothed_state_estimater)

        return pred_state


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


def Kalman2D_Pykalman(measurements):
    # code by Huangxunhua
    # input shape:(1600,2)

    filter = []

    initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf1 = KalmanFilter(transition_matrices=transition_matrix,
                       observation_matrices=observation_matrix,
                       initial_state_mean=initial_state_mean)

    kf1 = kf1.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

    dim1 = smoothed_state_means[:, 0]
    dim2 = smoothed_state_means[:, 2]

    filter.append(dim1)
    filter.append(dim2)
    filter = np.array(filter)

    return filter


def kalman2D_Git(data):
    P = np.identity(2)
    X = np.array([[0], [0]])
    dt = 5
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[dt * dt / 2], [dt]])
    Q = np.array([[.0001, .00002], [.00002, .0001]])
    R = np.array([[.01, .005], [.005, .02]])
    estimated = []
    H = np.identity(2)
    I = np.identity(2)
    # print("X")
    # print(X)
    for i in data:
        u1 = i[0]
        u2 = i[1]
        u_k = np.array([[u1], [u2]])
        u_k = np.squeeze(u_k)

        # z_k = np.squeeze(z_k)
        # prediction
        X = X + u_k
        P = A * P * A.T + Q
        # kalman gain/measurement
        K = P / (P + R)
        Y = np.dot(H, u_k).reshape(2, -1)

        # new X and P
        X = X + np.dot(K, Y - np.dot(H, X))
        P = (I - K * H) * P
        estimated.append(X)

    estimated = np.squeeze(np.array(estimated)[:, :, :1])

    return estimated


def Savitzky(x):
    x_sav = []
    print("SavitzkyFiltering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)

        signal_sav = savgol_filter(signal, 51, 5)

        # WavePlot_Single(signal_sav,'kalman')

        x_sav.append(signal_sav)

    x_sav = np.array(x_sav)
    x_sav = np.expand_dims(x_sav, 1)

    return x_sav, x_sav.shape[-1]


def Wiener(x):
    x_Wie = []
    print("WienerFiltering.....")

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
    x_Wie = np.expand_dims(x_Wie, 1)

    return x_Wie, x_Wie.shape[-1]


def Kalman_1D(x):
    x_Kal = []
    print("Kalman1D  Filtering.....")

    w_size = x.shape[1]
    if w_size % 2 == 0:
        w_size = w_size + 1

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            signal = np.squeeze(signal)

            # WavePlot_Single(x[i],'signal')

            # signal_sav = KalmanFilter(signal,len(signal))
            signal_Kalman = Kalman1D(signal)

            # WavePlot_Single(signal_sav,'kalman')

            x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]


def Kalman_2D(x):
    x_Kal = []
    print("Kalman2D  Filtering.....")
    x = x.transpose(0, 2, 1)

    # w_size = x.shape[1]
    # if w_size % 2 == 0:
    #     w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        print(i)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = Kalman2D_Pykalman(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    # x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]


def Kalman_2D_Git(x):
    x_Kal = []
    print("Kalman2D_Git  Filtering.....")
    x = x.transpose(0, 2, 1)

    # w_size = x.shape[1]
    # if w_size % 2 == 0:
    #     w_size = w_size + 1

    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        # print(i)

        # WavePlot_Single(x[i],'signal')

        # signal_sav = KalmanFilter(signal,len(signal))
        signal_Kalman = kalman2D_Git(signal)

        # WavePlot_Single(signal_sav,'kalman')

        x_Kal.append(signal_Kalman)

    x_Kal = np.array(x_Kal)
    x_Kal = x_Kal.transpose(0, 2, 1)
    # x_Kal = np.expand_dims(x_Kal, 1)

    return x_Kal, x_Kal.shape[-1]


# def KalmanFilter(z, n_iter=20):
#     # 这里是假设A=1，H=1的情况
#
#     # intial parameters
#
#     sz = (n_iter,)  # size of array
#
#     # Q = 1e-5 # process variance
#     Q = 1e-6  # process variance  变化量
#     # allocate space for arrays
#     xhat = np.zeros(sz)  # a posteri estimate of x    预测
#     P = np.zeros(sz)  # a posteri error estimate
#     xhatminus = np.zeros(sz)  # a priori estimate of x   观测
#     Pminus = np.zeros(sz)  # a priori error estimate
#     K = np.zeros(sz)  # gain or blending factor
#
#     R = 0.1 ** 2  # estimate of measurement variance, change to see effect
#
#     # intial guesses
#     xhat[0] = 0.0
#     P[0] = 1.0
#     A = 1
#     H = 1
#
#     for k in range(1, n_iter):
#         # time update
#         xhatminus[k] = A * xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0   观测 = 上一个预测
#         Pminus[k] = A * P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1    观测误差 = 预测误差+变化量   观测准确度
#
#         # measurement update
#         K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1   均化系数 = 观测误差    卡尔曼增益
#         xhat[k] = xhatminus[k] + K[k] * (z[k] - H * xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  预测 = 观测 + 卡尔曼增益*（真实-观测(上一个预测)）
#         P[k] = (1 - K[k] * H) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1   预测误差=（1-均化系数）*观测误差   省去了预测？
#
#     return xhat


def Gussian_Noisy(x, snr):  # snr:信噪比

    x_gussian = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        # sum = np.sum(signal ** 2)

        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        # test = np.random.randn(len(signal))
        # WavePlot_Single(test,'random')

        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        WavePlot_Single(gussian, 'gussian')

        x_gussian.append(x[i] + gussian)

    x_gussian = np.array(x_gussian)
    x_gussian = np.expand_dims(x_gussian, 1)

    return x_gussian, x_gussian.shape[-1]


def Gamma_Noisy_return_N(x, snr):  # x:信号 snr:信噪比

    print('Gamma')
    x_gamma = []
    x_gamma_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])  # signal
        # WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        gamma = np.random.gamma(shape=2, size=len(signal)) * np.sqrt(npower)  # noisy
        WavePlot_Single_2(gamma, 'gamma_5')

        x_gamma.append(x[i] + gamma)  # add
        WavePlot_Single(x[i] + gamma, 'gamma_5_add')
        x_gamma_only.append(gamma)

    x_gamma = np.array(x_gamma)
    x_gamma = np.expand_dims(x_gamma, 1)
    x_gamma_only = np.array(x_gamma_only)
    x_gamma_only = np.expand_dims(x_gamma_only, 1)

    return x_gamma_only, x_gamma, x_gamma.shape[-1]


def Rayleign_Noisy_return_N(x, snr):  # snr:信噪比

    print('Ralyeign')
    x_rayleign = []
    x_rayleign_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        rayleign = np.random.rayleigh(size=len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(rayleign, 'rayleigh_5')

        x_rayleign.append(x[i] + rayleign)
        WavePlot_Single(x[i] + rayleign, 'rayleigh_5_add')
        x_rayleign_only.append(rayleign)

    x_rayleign = np.array(x_rayleign)
    x_rayleign = np.expand_dims(x_rayleign, 1)
    x_rayleign_only = np.array(x_rayleign_only)
    x_rayleign_only = np.expand_dims(x_rayleign_only, 1)

    return x_rayleign_only, x_rayleign, x_rayleign.shape[-1]


def Exponential_Noisy_return_N(x, snr):  # snr:信噪比

    print("Exponential")
    x_exponential = []
    x_exponential_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        exponential = np.random.exponential(size=len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(exponential, 'exponential_5')

        x_exponential.append(x[i] + exponential)
        WavePlot_Single(x[i] + exponential, 'exponential_5_add')
        x_exponential_only.append(exponential)

    x_exponential = np.array(x_exponential)
    x_exponential = np.expand_dims(x_exponential, 1)
    x_exponential_only = np.array(x_exponential_only)
    x_exponential_only = np.expand_dims(x_exponential_only, 1)

    return x_exponential_only, x_exponential, x_exponential.shape[-1]


def Uniform_Noisy_return_N(x, snr):  # snr:信噪比

    print("Uniform")
    x_uniform = []
    x_uniform_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        uniform = np.random.uniform(size=len(signal)) * np.sqrt(npower)
        WavePlot_Single_2(uniform, 'uniform_5')

        x_uniform.append(x[i] + uniform)
        WavePlot_Single(x[i] + uniform, 'uniform_5_add')
        x_uniform_only.append(uniform)

    x_uniform = np.array(x_uniform)
    x_uniform = np.expand_dims(x_uniform, 1)
    x_uniform_only = np.array(x_uniform_only)
    x_uniform_only = np.expand_dims(x_uniform_only, 1)

    return x_uniform_only, x_uniform, x_uniform.shape[-1]


def Poisson_Noisy_return_N(x, snr):  # snr:信噪比

    print("possion")
    x_poisson = []
    x_poisson_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        i = 3

        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        poisson = np.random.poisson(1, len(signal)) * np.sqrt(npower)
        WavePlot_Single(poisson, 'poisson_5')

        x_poisson.append(x[i] + poisson)
        WavePlot_Single(x[i] + poisson, 'poisson_5_add')
        x_poisson_only.append(poisson)

    x_poisson = np.array(x_poisson)
    x_poisson = np.expand_dims(x_poisson, 1)
    x_poisson_only = np.array(x_poisson_only)
    x_poisson_only = np.expand_dims(x_poisson_only, 1)

    return x_poisson_only, x_poisson, x_poisson.shape[-1]


import time


def Gussian_Noisy_return_N(x, snr):  # snr:信噪比

    # x_gussian = []
    x_gussian = np.zeros_like(x)
    x_gussian_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        # i = 3
        for j in range(x.shape[1]):
            signal = np.array(x[i][j])
            # signal = np.array(x[i])
            signal = np.squeeze(signal)
            xpower = np.sum(signal ** 2) / len(signal)
            npower = xpower / snr
            gussian = np.random.randn(len(signal)) * np.sqrt(npower)
            # WavePlot_Single_2(gussian, 'gussian_5')

            x_gussian[i][j] = (signal + gussian)
            # x_gussian.append(x[i] + gussian)

            x_gussian_only.append(gussian)

    x_gussian = np.array(x_gussian)
    # x_gussian = np.expand_dims(x_gussian, 1)
    x_gussian_only = np.array(x_gussian_only)
    x_gussian_only = np.expand_dims(x_gussian_only, 1)

    return x_gussian_only, x_gussian, x_gussian.shape[-1]


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1


def getPercent(data_x, data_y, percent, seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=percent, random_state=seed)
    return train_x, test_x, train_y, test_y


def one_class_labeling(labels, normal_class: int, seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)

    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


def one_class_labeling_sz(labels, abnormal_class: int, seed):
    normal_idx = np.where(labels != abnormal_class)[0]
    abnormal_idx = np.where(labels == abnormal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


def one_class_labeling_multi(labels, normal_classes):
    all_idx = np.asarray(list(range(len(labels))))
    for normal_class in normal_classes:
        normal_idx = np.where(labels == normal_class)[0]

    abnormal_idx = np.delete(all_idx, normal_idx, axis=0)

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class WaveLetDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_wavelet, _ = wavelet_preprocessing_set(X)
        # X = np.expand_dims(X, 1)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_wavelet = torch.Tensor(X_wavelet)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_wavelet[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class SavitzkyDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_Sav, _ = Savitzky(X)
        X = np.expand_dims(X, 1)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Sav = torch.Tensor(X_Sav)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Sav[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class WienerDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        X_Wie, _ = Wiener(X)
        X = np.expand_dims(X, 1)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Wie = torch.Tensor(X_Wie)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Wie[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class KalmanDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """
        X_Kal, _ = Kalman_1D(X)
        X = np.expand_dims(X, 1)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        self.X = torch.Tensor(X)
        self.X_Kal = torch.Tensor(X_Kal)
        self.Y = torch.Tensor(Y)

    class KalmanDataset(data.Dataset):
        def __init__(self, X, Y):
            """
            """
            X_Kal, _ = Kalman_1D(X)
            X = np.expand_dims(X, 1)
            # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
            self.X = torch.Tensor(X)
            self.X_Kal = torch.Tensor(X_Kal)
            self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Kal[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class UCR2018_noise(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.expand_dims(img, 0)
        if self.transform is not None:
            img_transformed = self.transform(img.copy())
        else:
            img_transformed = img
        return img, img_transformed, target

    def __len__(self):
        return self.data.shape[0]


class cpsc2018_noise(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = np.expand_dims(img, 0)
        if self.transform is not None:
            img_transformed = self.transform(img.copy())
        else:
            img_transformed = img

        return img, img_transformed, target

    def __len__(self):
        return self.data.shape[0]


class LorentzDataset(data.Dataset):

    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        np.zeros_like(x)

        x_lr = []
        x_sdnn = []
        x_rmssd = []

        x_lr_1 = []
        x_lr_2 = []
        x_lr_3 = []

        x_sdnn_1 = []
        x_rmssd_1 = []

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                lr, sdnn, rmssd = get_nni_24H_1(x[i][j], 100)
                lr_1 = lr[0]
                lr_2 = lr[1]
                x_lr_1 = x_lr_1 + lr_1
                x_lr_2 = x_lr_2 + lr_2

                x_sdnn_1.append(sdnn)
                x_rmssd_1.append(rmssd)

            x_lr_1 += [0 for i in range(np.size(x, 2) - len(x_lr_1))]
            x_lr_2 += [0 for i in range(np.size(x, 2) - len(x_lr_2))]

            x_lr_3.append(x_lr_1)
            x_lr_3.append(x_lr_2)

            x_lr.append(x_lr_3)
            x_sdnn.append(x_sdnn_1)
            x_rmssd.append(x_rmssd_1)

            x_lr_1 = []
            x_lr_2 = []
            x_lr_3 = []
            x_sdnn_1 = []
            x_rmssd_1 = []

        x_lr = np.asarray(x_lr, dtype=np.float32)
        x_sdnn = np.asarray(x_sdnn, dtype=np.float32)
        x_rmssd = np.asarray(x_rmssd, dtype=np.float32)
        x_sdnn = np.mean(x_sdnn, 1)
        x_rmssd = np.mean(x_rmssd, 1)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_lr = torch.Tensor(x_lr)
        self.x_sdnn = torch.Tensor(x_sdnn)
        self.x_rmssd = torch.Tensor(x_rmssd)

    def __getitem__(self, index):

        return self.x[index], self.x_lr[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class GussianNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, X_gussian, _ = Gussian_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        # X = np.expand_dims(X, 1)

        self.X = torch.Tensor(X)
        self.X_Gussian = torch.Tensor(X_gussian)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Gussian[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GussianNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Gussain, _ = Gussian_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Nosiy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Nosiy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class PossionNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Possion, _ = Poisson_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Possion)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class PossionNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Possion, _ = Poisson_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class UniformNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Uniform, _ = Uniform_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Uniform)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class UniformNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Uniform, _ = Uniform_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class ExponentialNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Exponential, _ = Exponential_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Nosiy = torch.Tensor(Exponential)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Nosiy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class ExponentialNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Exponential, _ = Exponential_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class RayleignNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Rayleign, _ = Rayleign_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Rayleign)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class RayleignNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        Noisy, Rayleign, _ = Rayleign_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Noisy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Noisy_Only[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class GammaNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        _, Gamma, _ = Gamma_Noisy_return_N(X, SNR)
        # WavePlot(X_wavelet[0][0],X_wavelet[1][0],X_wavelet[2][0],X_wavelet[3][0])
        X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Add_Noisy = torch.Tensor(Gamma)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Add_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


def WavePlot(x1, x2, x3, x4):
    x = np.linspace(0, len(x1) * 4, len(x1) * 4)
    a = list(x1)
    b = list(x2)
    d = list(x3)
    e = list(x4)
    a.extend(b)
    a.extend(d)
    a.extend(e)
    c = np.array(a)

    y = c

    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    plt.legend()

    plt.show()
    plt.savefig("data_1.svg")


def WavePlot_Single(x1, name):
    x = np.linspace(0, len(x1), len(x1))
    a = list(x1)

    c = np.array(a)

    y = c

    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = (5, 3)
    # plt.legend()

    # plt.show()
    plt.savefig('{}.svg'.format(name))
    plt.close()


def WavePlot_Single_2(x1, name):
    x = np.linspace(0, len(x1), len(x1))
    a = list(x1)

    c = np.array(a)

    y = c

    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    plt.tight_layout()
    plt.rcParams['figure.figsize'] = (5, 1)
    # plt.legend()

    # plt.show()
    plt.savefig('{}.svg'.format(name))
    plt.close()


def preprocess_signals(X_train, X_validation, X_test, outputfolder=None):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    # with open(outputfolder + 'standard_scaler.pkl', 'wb') as ss_file:
    #   pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def get_EpilepticSeizure(dataset_path, dataset_name):
    data = []
    data_x = []
    data_y = []
    f = open('{}/{}/data.csv'.format(dataset_path, dataset_name), 'r')
    for line in range(0, 11501):
        if line == 0:
            f.readline()
            continue
        else:
            data.append(f.readline().strip())
    for i in range(0, 11500):
        tmp = data[i].split(",")
        del tmp[0]
        del tmp[178]
        data_x.append(tmp)
        data_y.append(data[i][-1])
    data_x = np.asfarray(data_x, dtype=np.float32)
    data_y = np.asarray([int(x) - 1 for x in data_y], dtype=np.int64)

    return data_x, data_y


def AD_data(dataset_name, data_X, data_Y, normal_idx, opt):
    # if dataset_name == 'EpilepticSeizure':
    #     labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)
    #
    # else:
    #     labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)

    labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)
    data_N_X = data_X[idx_normal]  # (4187,96)
    data_N_Y = labels_binary[idx_normal]  # (4187,)  1D
    data_A_X = data_X[idx_abnormal]  # (12450,96)
    data_A_Y = labels_binary[idx_abnormal]  # UCR end

    # Split normal samples
    n_normal = data_N_X.shape[0]
    train_X = data_N_X[:(int(n_normal * 0.6)), ]  # train 0.6
    train_Y = data_N_Y[:(int(n_normal * 0.6)), ]

    val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]  # train 0.2
    val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]

    test_N_X = data_N_X[int(n_normal * 0.8):]  # train 0.2
    test_N_Y = data_N_Y[int(n_normal * 0.8):]

    data_A_X_len = data_A_X.shape[0]

    val_N_X_len = data_A_X_len // 2
    test_N_X_len = data_A_X_len // 2

    data_A_X_idx = list(range(data_A_X_len))
    val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    val_X = np.concatenate((val_N_X, val_A_X))
    val_Y = np.concatenate((val_N_Y, val_A_Y))
    test_X = np.concatenate((test_N_X, test_A_X))
    test_Y = np.concatenate((test_N_Y, test_A_Y))
    return train_X, train_Y, val_X, val_Y, test_X, test_Y, val_N_X, val_A_X, test_N_X, test_A_X


def load_data_cpsc(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)

    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)
    NOT_ALL = False
    if NOT_ALL:
        train_X = train_data[:10000, :, :]
        train_Y = train_label[:10000]
    else:
        train_X = train_data
        train_Y = train_label

    val_X = val_data
    val_Y = val_label
    test_X = test_data
    test_Y = test_label

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.model in ['MM_ECG']:
        train_dataset = LorentzDataset(train_X, train_Y)
        val_dataset = LorentzDataset(val_X, val_Y)
        test_dataset = LorentzDataset(test_X, test_Y)

    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length
