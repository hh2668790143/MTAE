# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/11/18 20:28
# 导入库
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_2(signal, signal_noise, noise_label=0, save_dir=None, datename=None):
    LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    Noise_Label = ['Gussian', 'Uniform', 'Exponential', 'Rayleign', 'Gamma']
    try:
        # Plot
        signal = signal[:, :1000]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 2, 3, 4]
        for i in range(len(signal)):
            plt.subplot(4, 1, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        for i in range(len(signal_noise)):
            plt.subplot(4, 1, idx[i + 2])
            plt.plot(signal_noise[i], color='green', label=str(i))
            plt.title(LEAD[i] + "_" + Noise_Label[noise_label] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        # for i in range(len(rec_signal)):
        #     plt.subplot(6, 1, idx[i+4])
        #     plt.plot(rec_signal[i], color='green', label=str(i))
        #     plt.title(LEAD[i]+"_rec: ", fontsize=16)
        #     plt.xticks(x, x_labels)
        #     plt.xlabel('time (s)', fontsize=16)
        #     plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        if save_dir is not None and datename is not None:
            plt.savefig("{}/sample_rate_{}".format(save_dir, datename) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def plot_sample_3(signal, signal_noise, rec_signal, noise_label=0, save_dir=None, datename=None):
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
            plt.title(LEAD[i] + "_" + Noise_Label[noise_label] + ": ", fontsize=16)
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
        if save_dir is not None and datename is not None:
            plt.savefig("{}/sample_rate_{}".format(save_dir, datename) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


# signal=np.array(np.random.random((2,1000)))
# signal_noise=np.array(np.random.random((2,1000)))
# plot_sample_2(signal,signal_noise)
# print()


def plot_img(input, img_num: int, name, opt):
    """img:目标图像"""
    """input:输入训练集图像"""
    # print(name + " start")
    plt.figure(dpi=300, figsize=(24, 15))
    for i in range(img_num):
        plt.subplot(8, 3, i + 1)

        plt.plot(input.tolist()[i][0], color='red', label='input', linewidth=3)

    plt.legend(loc="best")

    plt.savefig('./Plot_img/{}/{}/{}_{}_{}.png'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.savefig('./Plot_img/{}/{}/{}_{}_{}.pdf'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.close()
    print('Plot')


def plot_img_ecg(rec_x, x, x_noise, img_num: int, name, opt):
    """img:目标图像"""
    """input:输入训练集图像"""
    print(name + " start")
    plt.figure(dpi=300, figsize=(24, 15))
    for i in range(img_num):
        plt.subplot(8, 3, i + 1)
        # plt.plot(rec_x.tolist()[i][0], color='green', linestyle='--', label='rec_x')
        plt.plot(x.tolist()[i][0], color="blue", linestyle=":", label='x')
        plt.plot(x_noise.tolist()[i][0], color="red", linestyle="-", label='x_noise')

    plt.legend(loc="best")
    plt.savefig("./img_noise/{}_{}_".format(opt.snr, opt.noise) + name + ".jpg")
    plt.close()
    print(name + " over")


def plot_img_3(img, input, label, name, opt):
    """img:目标图像"""
    """input:输入训练集图像"""
    print("start")

    fig, axs = plt.subplots(10, 8, dpi=100, figsize=(30, 30))
    axs = axs.flatten()

    # fig.subplots_adjust(wspace=0.4, hspace=0.6)
    for ax in axs:
        ax.axis('off')

    heat_rec = (np.array(img.tolist()) - np.array(input.tolist())) ** 2

    for i in [17]:
        ax_11 = axs[i].inset_axes([0.0, 0.26, 1, 0.74])
        if label.tolist()[i] == 0:
            color = 'blue'
            input_label = "Normality"
        else:
            color = 'red'
            input_label = "Anomaly"
        ax_11.plot(input.tolist()[i][0], color="red", linestyle='-', linewidth=1, label="Anomaly")
        ax_11.plot(input.tolist()[i][0], color="blue", linestyle='-', linewidth=1, label="Normality")
        ax_11.plot(img.tolist()[i][0], color="black", linestyle='--', linewidth=1, label="Reconstruction")

        # ax_11.tick_params(axis='both', which='both', color='none')
        # ax_11.set_xticks([])
        ax_11.set_yticks([])
        # ax_11.legend(loc='best')
        if i == 17:
            ax_11.legend(loc='upper left', bbox_to_anchor=(-0.0, 1.5), ncol=3, fontsize=10)

        ax_12 = axs[i].inset_axes([0.0, 0.13, 1, 0.10], sharex=ax_11)

        heat_norm = np.reshape(heat_norm, (1, -1))

        ax_13 = axs[i].inset_axes([0.0, 0.0, 1, 0.10], sharex=ax_11)
        heat_1 = (np.array(img.tolist()[i][0]) - np.array(input.tolist()[i][0])) ** 2
        heat_norm_1 = np.reshape(heat_norm_1, (1, -1))

        if np.max(heat_norm) > np.max(heat_norm_1):
            vmax = np.max(heat_norm)  # get the maximum value of heat_norm
        else:
            vmax = np.max(heat_norm_1)
        ax_12.imshow(heat_norm, cmap="jet", aspect="auto", vmin=0, vmax=vmax)
        ax_13.imshow(heat_norm_1, cmap="jet", aspect="auto", vmin=0, vmax=vmax)

        # ax_12.tick_params(axis='y', which='both')
        # ax_12.set_xticks([])
        ax_12.text(-0.03, 0.5, "Sim", transform=ax_12.transAxes, fontsize=10, va='center', ha='right')
        ax_12.set_yticks([])
        ax_13.set_yticks([])
        # ax_12.set_xticks([])
        ax_12.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_11.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_13.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax_13.text(-0.03, 0.5, "Rec", transform=ax_13.transAxes, fontsize=10, va='center', ha='right')

    plt.savefig('./Plot_img/{}/{}/{}_{}_{}.png'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.savefig('./Plot_img/{}/{}/{}_{}_{}.pdf'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.close()
    print('Plot')


def plot_img_4(img, input, label, name, opt):
    """img:目标图像"""
    """input:输入训练集图像"""
    print(name + " start")

    fig, axs = plt.subplots(10, 8, dpi=100, figsize=(30, 30))
    axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    heat_rec = (np.array(img.tolist()) - np.array(input.tolist())) ** 2

    for i in [17]:
        ax_11 = axs[i].inset_axes([0.0, 0.26, 1, 0.74])
        if label.tolist()[i] == 0:
            color = 'blue'
            input_label = "Normality"
        else:
            color = 'red'
            input_label = "Anomaly"

        # ax_11.plot(input.tolist()[i][0], color="red", linestyle='-', linewidth=1, label="Anomaly")
        # ax_11.plot(input.tolist()[i][0], color="blue", linestyle='-', linewidth=1, label="Normality")
        # ax_11.plot(img.tolist()[i][0], color="black", linestyle='--', linewidth=1, label="Reconstruction")
        ax_11.plot(input.tolist()[i][0], color="blue", linestyle='-', linewidth=1, label="Input")
        ax_11.plot(img.tolist()[i][0], color="black", linestyle='--', linewidth=1, label="Ouput")

        ax_11.set_yticks([])
        if i == 17:
            ax_11.legend(loc='upper left', bbox_to_anchor=(-0.0, 1.5), ncol=3, fontsize=10)

        ax_13 = axs[i].inset_axes([0.0, 0.0, 1, 0.10], sharex=ax_11)
        heat_1 = (np.array(img.tolist()[i][0]) - np.array(input.tolist()[i][0])) ** 2
        heat_norm_1 = (heat_1 - np.min(heat_rec)) / (np.max(heat_rec) - np.min(heat_rec))
        heat_norm_1 = np.reshape(heat_norm_1, (1, -1))

        vmax = np.max(heat_norm_1)

        ax_13.imshow(heat_norm_1, cmap="jet", aspect="auto", vmin=0, vmax=vmax)
        ax_13.set_yticks([])

        ax_11.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_13.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax_13.text(-0.03, 0.5, "Rec", transform=ax_13.transAxes, fontsize=10, va='center', ha='right')

    # plt.show()

    plt.savefig('./Plot_img1/{}/{}/{}_{}_{}.svg'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.savefig('./Plot_img1/{}/{}/{}_{}_{}.pdf'.format(opt.model, opt.dataset, opt.normal_idx, opt.seed, name),
                transparent=False,
                bbox_inches='tight')
    plt.close()
    print('Plot')


def save_ts_heatmap_1D(input, output, label, save_path, heat_normal):

    j = 0

    x_points = np.arange(input.shape[-1])
    # fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[8,2]})
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [6, 2]})

    sig_in = np.squeeze(input[j, :])
    sig_out = np.squeeze(output[j, :])
    label = label[j]

    # sig_out_add = sig_out+(np.round((sig_in[0]-sig_out[0]),1))
    # sig_out = sig_out+(np.mean(sig_in)-np.mean(sig_out))

    # 设置图例并且设置图例的字体及大小
    font1 = {'weight': 'normal',
             'size': 16, }

    ax[0].plot(x_points, sig_in, 'k--', linewidth=2.5, label="input signal")
    ax[0].plot(x_points, sig_out, 'k--', linewidth=2.5, color='blue', label="output signal")

    # ax[0].tick_params(labelsize=23)

    ax[0].set_yticks([])
    ax[0].set_xticks([])

    # zax[0].legend(loc="upper right", prop={'size': 23})

    # ax[0].patch.set_facecolor('white')

    heat = np.abs(sig_out - sig_in)  # 输出- 输入

    # heat = np.mean(heat,dim=1)

    for i in range(heat.shape[0]):

        if heat[i] < heat_normal:
            heat[i] = 0

    # heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))

    # for i in range(heat_norm.shape[0]):
    #
    #     if heat_norm[i] > 0.5:
    #
    #         heat_norm[i] = 1

    heat_norm = np.reshape(heat, (1, -1))

    ax[1].imshow(heat_norm, cmap="jet", aspect=8)
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    fig.tight_layout()

    # plt.xlabel('Channel:{}_iter:{}   Pre:{}'.format(0,epoch, "%.2f%%" % (pre * 100)),font1)  # X轴标签
    if int(label) == 0:
        plt.xlabel('Channel:{}    Label:{}'.format(0, 'Normal'), font1)  # X轴标签
    else:
        plt.xlabel('Channel:{}    Label:{}'.format(0, 'Abnormal'), font1)  # X轴标签

    # fig.show()
    # return
    fig.savefig('./Plot/{}.svg'.format(save_path))
    plt.clf()
    plt.close()
    plt.show()
