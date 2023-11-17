# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 10:20
# @Author  : Haoyi Fan
# @Email   : isfanhy@gmail.com
# @File    : utils.py
#from pyts.approximation import DiscreteFourierTransform


import numpy as np

def FFT(data):

    data_fft = []
    for x in data:
        x = (x - x.min()) / (x.max() - x.min())  # Data normalization

        x = np.fft.fft(x)

        x = np.abs(x) / len(x)
        # x = x[range(int(x.shape[0] / 2))]
        data_fft.append(x)
    data_fft = np.expand_dims(np.array(data_fft), 1)

    return data_fft, data_fft.shape[1:]

# piecewise aggregate approximation
def paa(data, ds_factor=2):
    # input datatype data : ndarray, dxn, d-number of series, n-number of samples in each series
    # input datatype ds_factor : int, downsampling factor, default is 2
    # output datatype ds_series : ndarray, downsampled time series

    d, ds_b = data.shape
    ds_length = int(ds_b / ds_factor)
    ds_series = np.empty(shape=(d, ds_length))
    for i in range(ds_length):
        #a = np.mean(data[:, i * ds_factor:(i + 1) * ds_factor], axis=1)
        ds_series[:, i] = np.mean(data[:, i * ds_factor:(i + 1) * ds_factor], axis=1).reshape(1,d)
    return ds_series.astype(np.float32)


# imaging time series as unthresholded recurrence plot
def r_plot(data, delay=0):
    # input datatype data : ndarray, 1xn, n-number of samples in each series
    # input datatype delay : int, delay embedding for RP formation, default value is 1
    # output datatype rp : ndarray, nxn, unthresholded recurrence plot for series

    transformed = np.zeros([2, len(data) - delay])
    transformed[0, :] = data[0:len(data) - delay]
    transformed[1, :] = data[delay:len(data)]
    rp = np.zeros([len(data) - delay, len(data) - delay])
    for i in range(len(rp)):
        temp = np.tile(transformed[:, i], (len(rp), 1)).T - transformed
        temp2 = np.square(temp)
        rp[i, :] = np.sum(temp2, axis=0)
    return np.array(rp)


# rescaling series into range [0,1]
def rescale(data):
    # input datatype data: ndarray , dxn, d-number of series, n-number of samples in each series
    # output datatype rescaled: ndarray, dxn

    num = data - np.tile(np.mat(data.min(axis=1)).T, (1, np.shape(data)[1]))
    denom = np.tile(np.mat(data.max(axis=1)).T, (1, np.shape(data)[1])) - np.tile(np.mat(data.min(axis=1)).T,
                                                                                  (1, np.shape(data)[1]))
    rescaled = np.multiply(num, 1 / denom)
    return rescaled


# imaging time series as Gramian Angular Difference Field (GADF)
def polar_rep(data):
    # input datatype data : ndarray, 1xn, n-number of samples in each series
    # output datatype phi : ndarray, 1xn
    # output datatype r : ndarray, 1xn

    phi = np.arccos(data)
    r = (np.arange(0, np.shape(data)[1]) / np.shape(data)[1]) + 0.1
    return phi, r


def GADF(data):
    # input datatype data : ndarray, 1xn, n-number of samples in each series
    # output datatype gadf : ndarray, nxn, GADF for series

    datacos = np.array(data)
    datasin = np.sqrt(1 - datacos ** 2)
    gadf = datasin.T * datacos - datacos.T * datasin
    return gadf


def DFT(train_ds, test_ds):
    # DFT transformation
    n_coefs = 30
    n_samples = train_ds.shape[0]
    n_timestamps = train_ds.shape[1]
    dft = DiscreteFourierTransform(n_coefs=n_coefs, norm_mean=False,
                                   norm_std=False)
    X_dft = dft.fit_transform(train_ds)
    # Compute the inverse transformation
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples,))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    train_dft = np.fft.irfft(X_dft_new, n_timestamps).astype(np.float32)


    n_coefs = 30
    n_samples = test_ds.shape[0]
    n_timestamps = test_ds.shape[1]
    X_dft = dft.fit_transform(test_ds)
    # Compute the inverse transformation
    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx],
                                            np.zeros((n_samples,))]
        ]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[
            X_dft[:, :1],
            X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]
        ]
    test_dft = np.fft.irfft(X_dft_new, n_timestamps).astype(np.float32)

    return train_dft, test_dft

