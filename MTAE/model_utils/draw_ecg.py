# -*- coding: utf-8 -*-
import os

# Author : chenpeng
# Time : 2022/11/18 20:28
# 导入库
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(signal):
    try:
        # Plot
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        plt.plot(signal, color='green')

        plt.xticks(x, x_labels)
        plt.xlabel('time (s)', fontsize=16)
        plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        # plt.savefig("Plot_Hist/11k_{}".format(9999) + '.svg', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False

def plot_sample1(signal, opt, label):

    signal = np.squeeze(signal)
    fig = plt.figure(figsize=(24, 3), dpi=300)
    x = np.arange(0, 1000, 100)
    x_labels = np.arange(0, 10)

    plt.plot(signal, color='cadetblue', label="II")
    plt.title("II" + ": ", fontsize=20, color='blue', loc='left')

    plt.xticks(x, x_labels)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('value (mV)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()

    save_dir = "./ECG_Simple/{}/{}_{}/{}".format(opt.datasetname, opt.section, label, opt.seed)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')
    # plt.savefig('{}/{}.svg'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')

    plt.show()
    plt.close()

    # plt.close()
    # print('Plot')


def plot_sample2(signal, opt, label, if_show=False):
    LEAD = ["I", "II"]
    try:
        signal = signal[:, :]
        fig = plt.figure(figsize=(18, 5), dpi=300)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)
        idx = [1, 2]
        for i in range(len(signal)):
            plt.subplot(2, 1, idx[i])
            plt.plot(signal[i], color='cadetblue', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=24, color='blue', loc='left')
            plt.xticks(x, x_labels)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('value (mV)', fontsize=24)

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)

        fig.tight_layout()

        if if_show:
            plt.show()
        else:
            save_dir = "./ECG_Simple/{}/{}_{}/{}".format(opt.dataset, opt.section, label, opt.seed)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # plt.savefig('{}/{}.pdf'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')
            # plt.savefig('{}/{}.png'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')
            plt.savefig('{}/{}.svg'.format(save_dir, str(opt.i)), transparent=False, bbox_inches='tight')

        plt.close()
        return True
    except Exception as e:
        print(e)
        return False


def plot_sample_3(signal, signal_noise, rec_signal, save_dir=None, datename=None):
    LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    Noise_Label = ['Gussian', 'Uniform', 'Exponential', 'Rayleign', 'Gamma']
    try:
        # Plot
        signal = signal[:, :1000]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 2, 3, 4, 5, 6]
        for i in range(len(signal)):
            plt.subplot(6, 1, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        for i in range(len(signal_noise)):
            plt.subplot(6, 1, idx[i + 2])
            plt.plot(signal_noise[i], color='green', label=str(i))
            plt.title(LEAD[i] + "_" + Noise_Label[0] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        for i in range(len(rec_signal)):
            plt.subplot(6, 1, idx[i + 4])
            plt.plot(rec_signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + "_rec: ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()

        plt.savefig("Plot_Hist/11k_{}".format(datename) + '.svg', bbox_inches='tight')
        # plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def plot_sample12(signal, save_dir=None, datename=None):
    LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    try:
        # Plot
        signal = signal[:, :]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        for i in range(len(signal)):
            plt.subplot(6, 2, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            # plt.title(LEAD[i] + ": ", fontsize=16)
            plt.title(LEAD[i], fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        plt.savefig("{}/sample_rate_{}".format(save_dir, datename) + '.png', bbox_inches='tight')
        # plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False

