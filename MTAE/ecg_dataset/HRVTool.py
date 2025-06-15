# -*- coding: utf-8 -*-

import sys

# Author : chenpeng
# Time : 2023/7/30 9:45
import numpy as np
import math
from pyhrv import tools

sys.path.append('/home/chenpeng/workspace/Noisy_MultiModal/experiments/')

import pyhrv.time_domain as td
import biosppy
import neurokit2 as nk
import os


def get_hr(signal_org, sp_rate=100):
    # hr_mean=0
    # hr_std=0
    # heart_rates=[0]
    # templates=[0]
    # rpeaks = [0]

    if len(signal_org.shape) > 1:
        signal_org = signal_org.squeeze()

    signal, rpeaks, templates_ts, templates, heart_rate_ts, heart_rates \
        = biosppy.signals.ecg.ecg(signal_org, sampling_rate=sp_rate, show=False)[1:]

    # templates = templates.tolist()
    rpeaks = rpeaks.tolist()

    # hr_mean = np.mean(heart_rates)
    # hr_std = np.std(heart_rates)
    # heart_rates = [int(hr) for hr in heart_rates]
    # return hr_mean, hr_std, heart_rates, templates, rpeaks
    return rpeaks


def get_nni(rpeaks, sp_rate=100):
    nni = [0]
    try:
        nni = tools.nn_intervals(rpeaks, sampling_rate=sp_rate)
    finally:
        return nni


def Gussian_Noisy(x, snr):  # snr:信噪比
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

    return x_gussian_only, x_gussian


def get_hrv(nni):
    sdnn, rmssd, nn20, pnn20, nn50, pnn50 = (0, 0, 0, 0, 0, 0)
    try:
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


def getRS(singal):
    sdnn = []
    rmssd = []
    afi = []
    nn20 = []
    pnn20 = []
    nn50 = []
    pnn50 = []
    err = 0
    for b, i in enumerate(singal):
        if b % 100 == 0:
            print("已处理: " + str(b))
        if len(i.shape) > 1:
            i = i.squeeze()
        try:
            if len(i.shape) > 1 and i.shape[0] > 1:
                rpeaks0 = get_hr(i[0])
                # _, rpeaks0 = nk.ecg_peaks(i[0], sampling_rate=100)

                nni0 = get_nni(rpeaks0)
                sdnn_1, rmssd_1, nn20_1, pnn20_1, nn50_1, pnn50_1 = get_hrv(nni0)
                if sdnn_1 == 0 and rmssd_1 == 0:
                    err += 1
                rpeaks1 = get_hr(i[1])
                # _, rpeaks1 = nk.ecg_peaks(i[1], sampling_rate=100)

                nni1 = get_nni(rpeaks1)


                sdnn_2, rmssd_2, nn20_2, pnn20_2, nn50_2, pnn50_2 = get_hrv(nni1)
                if sdnn_1 == 0 and rmssd_1 == 0:
                    err += 1
                mean_nn1 = np.mean(nni0)
                afi1 = sdnn_1 / mean_nn1
                mean_nn2 = np.mean(nni1)
                afi2 = sdnn_2 / mean_nn2

                # meds=[]
                # medr=[]
                # meds.append(sdnn_1)
                # meds.append(sdnn_2)
                # medr.append(rmssd_1)
                # medr.append(rmssd_2)
                # sdnn.append(meds)
                # rmssd.append(medr)
                sdnn.append((sdnn_1 + sdnn_2) / 2)
                rmssd.append((rmssd_1 + rmssd_2) / 2)
                afi.append((afi1 + afi2) / 2)
                # sdnn.append(max(sdnn_1, sdnn_2))
                # rmssd.append(max(rmssd_1,  rmssd_2))
            else:
                rpeaks = get_hr(i)
                # _, rpeaks = nk.ecg_peaks(i, sampling_rate=100)

                nni = get_nni(rpeaks)
                sdnn_1, rmssd_1, nn20_1, pnn20_1, nn50_1, pnn50_1 = get_hrv(nni)
                if sdnn_1 == 0 and rmssd_1 == 0:
                    err += 1
                mean_nn = np.mean(nni)
                afi.append(sdnn_1 / mean_nn)
                sdnn.append(sdnn_1)
                rmssd.append(rmssd_1)
        except Exception as e:
            print(e)
            err += 1
            sdnn.append(0)
            rmssd.append(0)
            afi.append(0)
            # if len(i.shape)>1 and i.shape[0] > 1:
            #     sdnn.append([0,0])
            #     rmssd.append([0,0])
            #     afi.append(0)
            # else:
            #     sdnn.append(0)
            #     rmssd.append(0)
            #     afi.append(0)
    print("无法计算：" + str(err))
    return np.array(sdnn), np.array(rmssd), np.array(afi)


def replace_list_segments(lst, segments, replacements):
    # 遍历替换段和替换内容
    for segment, replacement in zip(segments, replacements):
        # 使用切片操作，替换列表的指定段
        lst[segment[0]:segment[1]] = replacements[segment[0]:segment[1]]
    return lst


def replace_list2avg(lst, seg):
    for i in seg:
        lst[i[0]:i[1]] = [(lst[i[0]] + lst[i[1]]) / 2] * (i[1] - i[0])
    return lst


def getP(ecg_signal, datasetName, seed):
    res_ecg = []
    ecg_signal = ecg_signal.astype('float32')
    if datasetName in ['icentia11k']:
        if os.path.exists("icentia11k_" + str(seed) + ".npy"):
            res_ecg = np.load("icentia11k_" + str(seed) + ".npy")
        else:
            for index, i in enumerate(ecg_signal):
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                # x_gussian_only, x_gussian = Gussian_Noisy(i, snr)

                i = i.squeeze()
                # rpeaks = (nk.ecg_peaks(i, sampling_rate=100)[1]['ECG_R_Peaks'])
                try:
                    rpeaks = get_hr(i)
                except:
                    rpeaks = (nk.ecg_peaks(i, sampling_rate=100)[1]['ECG_R_Peaks'])
                _, waves_peak = nk.ecg_delineate(i, rpeaks, sampling_rate=100, method="dwt", show=False,
                                                 show_type='all')
                p_wave = waves_peak['ECG_P_Peaks']
                p_d = []
                for j in p_wave:
                    p_d.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                # x_gussian_only = x_gussian_only.squeeze()

                # ecg = replace_list_segments(i,p_d,np.zeros(1000))
                ecg = replace_list2avg(i, p_d)
                res_ecg.append(ecg)

            np.save("icentia11k_" + str(seed), np.array(res_ecg))
        return np.array(res_ecg)

    elif datasetName in ['cpsc2021']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])
                # rpeaks0 = get_hr(i[0])
                # rpeaks1 = get_hr(i[1])
                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']
                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                res_ecg.append([ecg0, ecg1])
            except:
                res_ecg.append([i[0], i[1]])

            np.save("cpsc2021_" + str(seed), np.array(res_ecg))
        print("MASK OK!")
        return np.array(res_ecg)

    elif datasetName in ['ptbxl']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                    rpeaks2 = get_hr(i[2])
                    rpeaks3 = get_hr(i[3])
                    rpeaks4 = get_hr(i[4])
                    rpeaks5 = get_hr(i[5])
                    rpeaks6 = get_hr(i[6])
                    rpeaks7 = get_hr(i[7])
                    rpeaks8 = get_hr(i[8])
                    rpeaks9 = get_hr(i[9])
                    rpeaks10 = get_hr(i[10])
                    rpeaks11 = get_hr(i[11])

                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks2 = (nk.ecg_peaks(i[2], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks3 = (nk.ecg_peaks(i[3], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks4 = (nk.ecg_peaks(i[4], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks5 = (nk.ecg_peaks(i[5], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks6 = (nk.ecg_peaks(i[6], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks7 = (nk.ecg_peaks(i[7], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks8 = (nk.ecg_peaks(i[8], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks9 = (nk.ecg_peaks(i[9], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks10 = (nk.ecg_peaks(i[10], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks11 = (nk.ecg_peaks(i[11], sampling_rate=100)[1]['ECG_R_Peaks'])

                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak2 = nk.ecg_delineate(i[2], rpeaks2, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak3 = nk.ecg_delineate(i[3], rpeaks3, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak4 = nk.ecg_delineate(i[4], rpeaks4, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak5 = nk.ecg_delineate(i[5], rpeaks5, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak6 = nk.ecg_delineate(i[6], rpeaks6, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak7 = nk.ecg_delineate(i[7], rpeaks7, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak8 = nk.ecg_delineate(i[8], rpeaks8, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak9 = nk.ecg_delineate(i[9], rpeaks9, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak10 = nk.ecg_delineate(i[10], rpeaks10, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')
                _, waves_peak11 = nk.ecg_delineate(i[11], rpeaks11, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')

                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']
                p_wave2 = waves_peak2['ECG_P_Peaks']
                p_wave3 = waves_peak3['ECG_P_Peaks']
                p_wave4 = waves_peak4['ECG_P_Peaks']
                p_wave5 = waves_peak5['ECG_P_Peaks']
                p_wave6 = waves_peak6['ECG_P_Peaks']
                p_wave7 = waves_peak7['ECG_P_Peaks']
                p_wave8 = waves_peak8['ECG_P_Peaks']
                p_wave9 = waves_peak9['ECG_P_Peaks']
                p_wave10 = waves_peak10['ECG_P_Peaks']
                p_wave11 = waves_peak11['ECG_P_Peaks']
                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d2 = []
                for j in p_wave2:
                    p_d2.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d3 = []
                for j in p_wave3:
                    p_d3.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d4 = []
                for j in p_wave4:
                    p_d4.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d5 = []
                for j in p_wave5:
                    p_d5.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d6 = []
                for j in p_wave6:
                    p_d6.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d7 = []
                for j in p_wave7:
                    p_d7.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d8 = []
                for j in p_wave8:
                    p_d8.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d9 = []
                for j in p_wave9:
                    p_d9.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d10 = []
                for j in p_wave10:
                    p_d10.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d11 = []
                for j in p_wave11:
                    p_d11.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                ecg2 = replace_list2avg(i[2], p_d2)
                ecg3 = replace_list2avg(i[3], p_d3)
                ecg4 = replace_list2avg(i[4], p_d4)
                ecg5 = replace_list2avg(i[5], p_d5)
                ecg6 = replace_list2avg(i[6], p_d6)
                ecg7 = replace_list2avg(i[7], p_d7)
                ecg8 = replace_list2avg(i[8], p_d8)
                ecg9 = replace_list2avg(i[9], p_d9)
                ecg10 = replace_list2avg(i[10], p_d10)
                ecg11 = replace_list2avg(i[11], p_d11)
                res_ecg.append([ecg0, ecg1, ecg2, ecg3, ecg4, ecg5, ecg6, ecg7, ecg8, ecg9, ecg10, ecg11])
            except:
                res_ecg.append([i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]])

            np.save("ptbxl_" + str(seed), np.array(res_ecg))

        print("MASK OK!")
        return np.array(res_ecg)


def getP_s(ecg_signal, datasetName, seed, section):
    res_ecg = []
    ecg_signal = ecg_signal.astype('float32')
    if datasetName in ['icentia11k']:

        for index, i in enumerate(ecg_signal):
            if index % 100 == 0:
                print("已处理：" + str(index))

            i = i.squeeze()

            try:
                rpeaks = get_hr(i)
            except:
                rpeaks = (nk.ecg_peaks(i, sampling_rate=100)[1]['ECG_R_Peaks'])
            _, waves_peak = nk.ecg_delineate(i, rpeaks, sampling_rate=100, method="dwt", show=False,
                                             show_type='all')
            p_wave = waves_peak['ECG_P_Peaks']
            p_wave = [x for x in p_wave if not math.isnan(x)]
            p_d = []
            for j in p_wave:
                p_d.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

            ecg = replace_list2avg(i, p_d)
            res_ecg.append(ecg)

        res_ecg = np.expand_dims(res_ecg, axis=1)

        if not os.path.exists(
                "./AFPD/{}/{}/{}".format('mask', datasetName, section)):
            os.makedirs("./AFPD/{}/{}/{}".format('mask', datasetName, section))

        np.save("./AFPD/{}/{}/{}/{}.npy".format('mask', datasetName, section, seed), np.array(res_ecg))

    elif datasetName in ['cpsc2021']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])

                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']

                p_wave0 = [x for x in p_wave0 if not math.isnan(x)]
                p_wave1 = [x for x in p_wave1 if not math.isnan(x)]

                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                res_ecg.append([ecg0, ecg1])
            except:
                res_ecg.append([i[0], i[1]])

        if not os.path.exists(
                "./AFPD/{}/{}/{}".format('mask', datasetName, section)):
            os.makedirs("./AFPD/{}/{}/{}".format('mask', datasetName, section))

        np.save("./AFPD/{}/{}/{}/{}.npy".format('mask', datasetName, section, seed), np.array(res_ecg))

    elif datasetName in ['ptbxl']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                    rpeaks2 = get_hr(i[2])
                    rpeaks3 = get_hr(i[3])
                    rpeaks4 = get_hr(i[4])
                    rpeaks5 = get_hr(i[5])
                    rpeaks6 = get_hr(i[6])
                    rpeaks7 = get_hr(i[7])
                    rpeaks8 = get_hr(i[8])
                    rpeaks9 = get_hr(i[9])
                    rpeaks10 = get_hr(i[10])
                    rpeaks11 = get_hr(i[11])

                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks2 = (nk.ecg_peaks(i[2], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks3 = (nk.ecg_peaks(i[3], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks4 = (nk.ecg_peaks(i[4], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks5 = (nk.ecg_peaks(i[5], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks6 = (nk.ecg_peaks(i[6], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks7 = (nk.ecg_peaks(i[7], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks8 = (nk.ecg_peaks(i[8], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks9 = (nk.ecg_peaks(i[9], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks10 = (nk.ecg_peaks(i[10], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks11 = (nk.ecg_peaks(i[11], sampling_rate=100)[1]['ECG_R_Peaks'])

                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak2 = nk.ecg_delineate(i[2], rpeaks2, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak3 = nk.ecg_delineate(i[3], rpeaks3, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak4 = nk.ecg_delineate(i[4], rpeaks4, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak5 = nk.ecg_delineate(i[5], rpeaks5, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak6 = nk.ecg_delineate(i[6], rpeaks6, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak7 = nk.ecg_delineate(i[7], rpeaks7, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak8 = nk.ecg_delineate(i[8], rpeaks8, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak9 = nk.ecg_delineate(i[9], rpeaks9, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak10 = nk.ecg_delineate(i[10], rpeaks10, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')
                _, waves_peak11 = nk.ecg_delineate(i[11], rpeaks11, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')

                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']
                p_wave2 = waves_peak2['ECG_P_Peaks']
                p_wave3 = waves_peak3['ECG_P_Peaks']
                p_wave4 = waves_peak4['ECG_P_Peaks']
                p_wave5 = waves_peak5['ECG_P_Peaks']
                p_wave6 = waves_peak6['ECG_P_Peaks']
                p_wave7 = waves_peak7['ECG_P_Peaks']
                p_wave8 = waves_peak8['ECG_P_Peaks']
                p_wave9 = waves_peak9['ECG_P_Peaks']
                p_wave10 = waves_peak10['ECG_P_Peaks']
                p_wave11 = waves_peak11['ECG_P_Peaks']

                p_wave0 = [x for x in p_wave0 if not math.isnan(x)]
                p_wave1 = [x for x in p_wave1 if not math.isnan(x)]
                p_wave2 = [x for x in p_wave2 if not math.isnan(x)]
                p_wave3 = [x for x in p_wave3 if not math.isnan(x)]
                p_wave4 = [x for x in p_wave4 if not math.isnan(x)]
                p_wave5 = [x for x in p_wave5 if not math.isnan(x)]
                p_wave6 = [x for x in p_wave6 if not math.isnan(x)]
                p_wave7 = [x for x in p_wave7 if not math.isnan(x)]
                p_wave8 = [x for x in p_wave8 if not math.isnan(x)]
                p_wave9 = [x for x in p_wave9 if not math.isnan(x)]
                p_wave10 = [x for x in p_wave10 if not math.isnan(x)]
                p_wave11 = [x for x in p_wave11 if not math.isnan(x)]

                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d2 = []
                for j in p_wave2:
                    p_d2.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d3 = []
                for j in p_wave3:
                    p_d3.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d4 = []
                for j in p_wave4:
                    p_d4.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d5 = []
                for j in p_wave5:
                    p_d5.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d6 = []
                for j in p_wave6:
                    p_d6.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d7 = []
                for j in p_wave7:
                    p_d7.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d8 = []
                for j in p_wave8:
                    p_d8.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d9 = []
                for j in p_wave9:
                    p_d9.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d10 = []
                for j in p_wave10:
                    p_d10.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d11 = []
                for j in p_wave11:
                    p_d11.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                ecg2 = replace_list2avg(i[2], p_d2)
                ecg3 = replace_list2avg(i[3], p_d3)
                ecg4 = replace_list2avg(i[4], p_d4)
                ecg5 = replace_list2avg(i[5], p_d5)
                ecg6 = replace_list2avg(i[6], p_d6)
                ecg7 = replace_list2avg(i[7], p_d7)
                ecg8 = replace_list2avg(i[8], p_d8)
                ecg9 = replace_list2avg(i[9], p_d9)
                ecg10 = replace_list2avg(i[10], p_d10)
                ecg11 = replace_list2avg(i[11], p_d11)
                res_ecg.append([ecg0, ecg1, ecg2, ecg3, ecg4, ecg5, ecg6, ecg7, ecg8, ecg9, ecg10, ecg11])
            except:
                res_ecg.append([i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]])

        if not os.path.exists(
                "./AFPD/{}/{}/{}".format('mask', datasetName, section)):
            os.makedirs("./AFPD/{}/{}/{}".format('mask', datasetName, section))

        np.save("./AFPD/{}/{}/{}/{}.npy".format('mask', datasetName, section, seed), np.array(res_ecg))

    return np.array(res_ecg)


def getP_test(ecg_signal, datasetName):
    res_ecg = []
    ecg_signal = ecg_signal.astype('float32')
    if datasetName in ['icentia11k']:

        for index, i in enumerate(ecg_signal):
            if index % 1000 == 0:
                print("已处理：" + str(index))

            i = i.squeeze()

            try:
                rpeaks = get_hr(i)
            except:
                rpeaks = (nk.ecg_peaks(i, sampling_rate=100)[1]['ECG_R_Peaks'])
            _, waves_peak = nk.ecg_delineate(i, rpeaks, sampling_rate=100, method="dwt", show=False,
                                             show_type='all')
            p_wave = waves_peak['ECG_P_Peaks']
            p_d = []
            for j in p_wave:
                p_d.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

            ecg = replace_list2avg(i, p_d)
            res_ecg.append(ecg)


    elif datasetName in ['cpsc2021']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])

                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']

                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                res_ecg.append([ecg0, ecg1])
            except:
                res_ecg.append([i[0], i[1]])


    elif datasetName in ['ptbxl']:

        for index, i in enumerate(ecg_signal):
            try:
                if index % 1000 == 0:
                    print("已处理：" + str(index))
                i = i.squeeze()
                try:
                    rpeaks0 = get_hr(i[0])
                    rpeaks1 = get_hr(i[1])
                    rpeaks2 = get_hr(i[2])
                    rpeaks3 = get_hr(i[3])
                    rpeaks4 = get_hr(i[4])
                    rpeaks5 = get_hr(i[5])
                    rpeaks6 = get_hr(i[6])
                    rpeaks7 = get_hr(i[7])
                    rpeaks8 = get_hr(i[8])
                    rpeaks9 = get_hr(i[9])
                    rpeaks10 = get_hr(i[10])
                    rpeaks11 = get_hr(i[11])

                except:
                    rpeaks0 = (nk.ecg_peaks(i[0], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks1 = (nk.ecg_peaks(i[1], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks2 = (nk.ecg_peaks(i[2], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks3 = (nk.ecg_peaks(i[3], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks4 = (nk.ecg_peaks(i[4], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks5 = (nk.ecg_peaks(i[5], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks6 = (nk.ecg_peaks(i[6], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks7 = (nk.ecg_peaks(i[7], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks8 = (nk.ecg_peaks(i[8], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks9 = (nk.ecg_peaks(i[9], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks10 = (nk.ecg_peaks(i[10], sampling_rate=100)[1]['ECG_R_Peaks'])
                    rpeaks11 = (nk.ecg_peaks(i[11], sampling_rate=100)[1]['ECG_R_Peaks'])

                _, waves_peak0 = nk.ecg_delineate(i[0], rpeaks0, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak1 = nk.ecg_delineate(i[1], rpeaks1, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak2 = nk.ecg_delineate(i[2], rpeaks2, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak3 = nk.ecg_delineate(i[3], rpeaks3, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak4 = nk.ecg_delineate(i[4], rpeaks4, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak5 = nk.ecg_delineate(i[5], rpeaks5, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak6 = nk.ecg_delineate(i[6], rpeaks6, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak7 = nk.ecg_delineate(i[7], rpeaks7, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak8 = nk.ecg_delineate(i[8], rpeaks8, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak9 = nk.ecg_delineate(i[9], rpeaks9, sampling_rate=100, method="dwt", show=False,
                                                  show_type='all')
                _, waves_peak10 = nk.ecg_delineate(i[10], rpeaks10, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')
                _, waves_peak11 = nk.ecg_delineate(i[11], rpeaks11, sampling_rate=100, method="dwt", show=False,
                                                   show_type='all')

                p_wave0 = waves_peak0['ECG_P_Peaks']
                p_wave1 = waves_peak1['ECG_P_Peaks']
                p_wave2 = waves_peak2['ECG_P_Peaks']
                p_wave3 = waves_peak3['ECG_P_Peaks']
                p_wave4 = waves_peak4['ECG_P_Peaks']
                p_wave5 = waves_peak5['ECG_P_Peaks']
                p_wave6 = waves_peak6['ECG_P_Peaks']
                p_wave7 = waves_peak7['ECG_P_Peaks']
                p_wave8 = waves_peak8['ECG_P_Peaks']
                p_wave9 = waves_peak9['ECG_P_Peaks']
                p_wave10 = waves_peak10['ECG_P_Peaks']
                p_wave11 = waves_peak11['ECG_P_Peaks']
                p_d0 = []
                for j in p_wave0:
                    p_d0.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d1 = []
                for j in p_wave1:
                    p_d1.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d2 = []
                for j in p_wave2:
                    p_d2.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d3 = []
                for j in p_wave3:
                    p_d3.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d4 = []
                for j in p_wave4:
                    p_d4.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d5 = []
                for j in p_wave5:
                    p_d5.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d6 = []
                for j in p_wave6:
                    p_d6.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d7 = []
                for j in p_wave7:
                    p_d7.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d8 = []
                for j in p_wave8:
                    p_d8.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d9 = []
                for j in p_wave9:
                    p_d9.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d10 = []
                for j in p_wave10:
                    p_d10.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])
                p_d11 = []
                for j in p_wave11:
                    p_d11.append([j - 5 if j - 5 >= 0 else 0, j + 5 if j + 5 < 1000 else 999])

                ecg0 = replace_list2avg(i[0], p_d0)
                ecg1 = replace_list2avg(i[1], p_d1)
                ecg2 = replace_list2avg(i[2], p_d2)
                ecg3 = replace_list2avg(i[3], p_d3)
                ecg4 = replace_list2avg(i[4], p_d4)
                ecg5 = replace_list2avg(i[5], p_d5)
                ecg6 = replace_list2avg(i[6], p_d6)
                ecg7 = replace_list2avg(i[7], p_d7)
                ecg8 = replace_list2avg(i[8], p_d8)
                ecg9 = replace_list2avg(i[9], p_d9)
                ecg10 = replace_list2avg(i[10], p_d10)
                ecg11 = replace_list2avg(i[11], p_d11)
                res_ecg.append([ecg0, ecg1, ecg2, ecg3, ecg4, ecg5, ecg6, ecg7, ecg8, ecg9, ecg10, ecg11])
            except:
                res_ecg.append([i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]])


    return np.array(res_ecg)