import multiprocessing
import time

import numpy as np
from hrvanalysis import get_time_domain_features
from matplotlib import pyplot as plt

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


def get_grv_list(signal_org, sample_rate):
    hr_mean, hr_std, heart_rates, templates, rpeaks = get_hr(signal_org, sample_rate)
    nni = get_nni(rpeaks, sample_rate)

    try:
        time_domain_features = get_time_domain_features(nni)
    except ValueError:
        hrv_list = [0 for i in range(30)]
    else:
        mean_nni = int(time_domain_features["mean_nni"])
        sdnn = int(time_domain_features["sdnn"])
        sdsd = int(time_domain_features["sdsd"])
        rmssd = int(time_domain_features["rmssd"])
        median_nni = int(time_domain_features["median_nni"])
        range_nni = int(time_domain_features["range_nni"])
        mean_hr = int(time_domain_features["mean_hr"])
        max_hr = int(time_domain_features["max_hr"])
        min_hr = int(time_domain_features["min_hr"])

        mean_nni = format(mean_nni, '04d')
        sdnn = format(sdnn, '03d')
        sdsd = format(sdsd, '03d')
        rmssd = format(rmssd, '03d')
        median_nni = format(median_nni, '04d')
        range_nni = format(range_nni, '04d')
        mean_hr = format(mean_hr, '03d')
        max_hr = format(max_hr, '03d')
        min_hr = format(min_hr, '03d')

        hrv_list = mean_nni + sdnn + sdsd + rmssd + median_nni + range_nni + mean_hr + max_hr + min_hr
        hrv_list = list(hrv_list)

    # print(hrv_list)

    return hrv_list


def draw_LR(x, y):
    plt.figure(figsize=(10, 10), dpi=100)
    plt.xlabel("nn1")
    plt.ylabel("nn2")

    mean1 = sum(x) / len(x)
    mean2 = sum(y) / len(y)
    mean = mean1 + mean2

    # max_ticks = max(max(x), max(y))
    my_x_ticks = np.arange(0, 200, 5)
    plt.xticks(my_x_ticks)
    my_y_ticks = np.arange(0, 200, 5)
    plt.yticks(my_y_ticks)
    plt.xlim(0, mean)
    plt.ylim(0, mean)
    plt.scatter(x, y)
    plt.show()


def plot_sample(signal, pp, n_p):
    try:
        # Plot
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        plt.plot(signal, color='green')
        pp_x = pp
        pp_y = [signal[i] for i in pp]

        np_x = n_p
        np_y = [signal[i] for i in n_p]
        plt.scatter(pp_x, pp_y, marker='o', color='red')
        plt.scatter(np_x, np_y, marker='*', color='blue')

        plt.xticks(x, x_labels)
        plt.xlabel('time (s)', fontsize=16)
        plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def replace_list_segments(lst, segments, replacements):
    # 遍历替换段和替换内容
    for segment, replacement in zip(segments, replacements):
        lst[segment[0]:segment[1]] = replacements[segment[0]:segment[1]]
    return lst
