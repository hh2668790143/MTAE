# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/12/4 20:26
import numpy as np
from MlultModal.draw_ECG.draw_ecg import plot_sample_2,plot_sample_3
from MlultModal.options import Options


def Gamma_Noisy_return_N(x, snr):  # snr:信噪比

    # print('Gamma')
    x_gamma = []
    x_gamma_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        # WavePlot_Single(signal, 'signal')
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
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

    # print('Ralyeign')
    x_rayleign = []
    x_rayleign_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
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

    # print("Exponential")
    x_exponential = []
    x_exponential_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
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

    # print("Uniform")
    x_uniform = []
    x_uniform_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        uniform = np.random.uniform(size=len(signal)) * np.sqrt(npower)
        # WavePlot_Single(uniform, 'uniform')

        x_uniform.append(x[i] + uniform)
        x_uniform_only.append(uniform)

    x_uniform = np.array(x_uniform)
    x_uniform_only = np.array(x_uniform_only)
    # x_uniform_only = np.expand_dims(x_uniform_only, 1)

    return x_uniform_only, x_uniform, x_uniform.shape[-1]

def Gussian_Noisy_return_N(x, snr):  # snr:信噪比
    x_gussian = []
    x_gussian_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)

        npower = xpower / snr
        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        # WavePlot_Single(gussian, 'gussian_200')

        x_gussian.append(x[i] + gussian)
        x_gussian_only.append(gussian)

    x_gussian = np.array(x_gussian)
    x_gussian_only = np.array(x_gussian_only)

    return x_gussian_only, x_gussian, x_gussian.shape[-1]


if __name__ == '__main__':
    from MlultModal.tool.CPSC.load_data_cpsc2021 import load_data,data_2_dataset
    root = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/"
    save = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/"
    opt = Options().parse()
    opt.seed=1
    opt.normalize=True

    opt.model="AE_CNN_self"
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)
    # self.X[index], self.fix[index], self.Y[index], self.Nosiy_Only[index], self.Nosiy_label[index]
    # save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
    # datename=str(epoch)+str(self.opt.seed)+str(self.opt.Snr)
    np.random.shuffle(train_data)
    j=0
    for z in range(20):
        j+=5
        print(j)
        for i in range(train_data.shape[0])[:10]:
            print(i)
            noise1, fix1, _ = Gussian_Noisy_return_N(train_data[i], j)
            noise2, fix2, _ = Uniform_Noisy_return_N(train_data[i], j)
            noise3, fix3, _ = Exponential_Noisy_return_N(train_data[i],j)
            noise4, fix4, _ = Rayleign_Noisy_return_N(train_data[i], j)
            noise5, fix5, _ = Gamma_Noisy_return_N(train_data[i], j)
            # Noise_Label=['Gussian','Uniform','Exponential','Rayleign','Gamma']
            plot_sample_3(train_data[i],fix1,noise1,
                          noise_label=0,
                          save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
                          datename="Gussian"+str(j)+"_"+str(i))
            plot_sample_3(train_data[i], fix2, noise2,
                          noise_label=1,
                          save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
                          datename="Uniform" + str(j) + "_" + str(i))
            plot_sample_3(train_data[i], fix3, noise3,
                          noise_label=2,
                          save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
                          datename="Exponential" + str(j) + "_" + str(i))
            plot_sample_3(train_data[i], fix4, noise4,
                          noise_label=3,
                          save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
                          datename="Rayleign" + str(j) + "_" + str(i))
            plot_sample_3(train_data[i], fix5, noise5,
                          noise_label=4,
                          save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
                          datename="Gamma" + str(j) + "_" + str(i))
    # for i in range(train_data.shape[0])[:10]:
    #     print(i)
    #     noise1, fix1, _ = Gussian_Noisy_return_N(train_data[i], 100)
    #     noise2, fix2, _ = Uniform_Noisy_return_N(train_data[i], 250)
    #     noise3, fix3, _ = Exponential_Noisy_return_N(train_data[i], 300)
    #     noise4, fix4, _ = Rayleign_Noisy_return_N(train_data[i], 350)
    #     noise5, fix5, _ = Gamma_Noisy_return_N(train_data[i], 350)
    #     plot_sample_3(train_data[i], fix1, noise1,
    #                   noise_label=0,
    #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img_new',
    #                   datename="Gussian" + str(i))
    #     plot_sample_3(train_data[i], fix2, noise2,
    #                   noise_label=1,
    #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img_new',
    #                   datename="Uniform" + str(i))
    #     plot_sample_3(train_data[i], fix3, noise3,
    #                   noise_label=2,
    #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img_new',
    #                   datename="Exponential" + str(i))
    #     plot_sample_3(train_data[i], fix4, noise4,
    #                   noise_label=3,
    #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img_new',
    #                   datename="Rayleign" + str(i))
    #     plot_sample_3(train_data[i], fix5, noise5,
    #                   noise_label=4,
    #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img_new',
    #                   datename="Gamma" + str(i))