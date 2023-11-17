import multiprocessing
import time

import numpy as np
import pyhrv.tools as tools
import pyhrv.time_domain as td
import biosppy


def get_hr(signal_org, sp_rate):
    hr_mean = 0
    hr_std = 0
    heart_rates = [0]
    templates = [0]
    rpeaks = [0]

    try:
        if len(signal_org.shape) > 1:
            signal_org = signal_org.squeeze()

        signal, rpeaks, templates_ts, templates, heart_rate_ts, heart_rates \
            = biosppy.signals.ecg.ecg(signal_org, sampling_rate=sp_rate, show=False)[1:]

        templates = templates.tolist()
        rpeaks = rpeaks.tolist()

        hr_mean = np.mean(heart_rates)
        hr_std = np.std(heart_rates)
        heart_rates = [int(hr) for hr in heart_rates]
    finally:
        return hr_mean, hr_std, heart_rates, templates, rpeaks


def get_nni(rpeaks, sp_rate=100):
    nni = [0]
    try:
        nni = tools.nn_intervals(rpeaks, sampling_rate=sp_rate)
    finally:
        return nni


def get_hrv(nni):
    sdnn, rmssd, nn20, pnn20, nn50, pnn50 = (0, 0, 0, 0, 0, 0)
    try:
        # Compute SDNN
        # results = td.time_domain(nni)
        sdnn_ = td.sdnn(nni)
        rmssd_ = td.rmssd(nni)
        nn20_ = td.nn20(nni)
        nn50_ = td.nn50(nni)

        sdnn = round(sdnn_['sdnn'], 1)
        rmssd = round(rmssd_['rmssd'], 1)
        nn20 = nn20_['nn20']
        pnn20 = round(nn20_['pnn20'], 1)
        nn50 = nn50_['nn50']
        pnn50 = round(nn50_['pnn50'], 1)
    finally:
        return sdnn, rmssd, nn20, pnn20, nn50, pnn50


def get_HRV(signal_org, sp_rate):
    hr_mean, hr_std, heart_rates, templates, rpeaks = get_hr(signal_org, sp_rate)
    nni = get_nni(rpeaks, sp_rate)
    sdnn, rmssd, nn20, pnn20, nn50, pnn50 = get_hrv(nni)
    return sdnn, rmssd


def get_nni_24H_1(signal_org, sample_rate):
    hr_mean, hr_std, heart_rates, templates, rpeaks = get_hr(signal_org, sample_rate)
    nni = get_nni(rpeaks, sample_rate)
    sdnn, rmssd, nn20, pnn20, nn50, pnn50 = get_hrv(nni)

    nni_24H = tools.nn_intervals(rpeaks)

    # nni_24H = nni_24H[nni_24H > 300]  # 去掉nn小于300ms的噪声点

    nni_24H_P = nni_24H[:-1]
    nni_24H_N = nni_24H[1:]

    nni_24H_P_N_list = []  # 洛伦兹散点图
    nni_24H_Single = []
    for i in range(int(nni_24H_P.size)):
        nni_24H_Single.append(float(nni_24H_P[i]))
        nni_24H_Single.append(float(nni_24H_N[i]))

        nni_24H_P_N_list.append(nni_24H_Single)
        nni_24H_Single = []

    nni_24H_P_N_list = list(map(list, zip(*nni_24H_P_N_list)))

    if nni_24H_P_N_list == []:
        nni_24H_P_N_list = [[0], [0]]
    # do_hist(heart_rates_24H)

    return nni_24H_P_N_list, sdnn, rmssd
