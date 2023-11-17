import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
# from dataset.transformer import r_plot, paa, rescale
# import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
import pywt
# import librosa
import scipy
# import yaml
import matplotlib.pyplot as plt
import pickle


def Gussian_Noisy(x, snr):  # snr:信噪比

    x_gussian = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        # WavePlot_Single(signal,'signal')
        signal = np.squeeze(signal)
        # sum = np.sum(signal ** 2)

        xpower = np.sum(signal ** 2) / len(signal)
        npower = xpower / snr
        # test = np.random.randn(len(signal))
        # WavePlot_Single(test,'random')

        gussian = np.random.randn(len(signal)) * np.sqrt(npower)
        #
        # plt.hist(gussian)
        # plt.savefig('hist')
        # WavePlot_Single(gussian,'gussian')
        x_gussian.append(x[i] + gussian)
        # WavePlot_Single(x[i]+gussian,'add')

    x_gussian = np.array(x_gussian)
    x_gussian = np.expand_dims(x_gussian, 1)

    return x_gussian, x_gussian.shape[-1]


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


def Poisson_Noisy_return_N(x, snr):  # snr:信噪比

    # print("possion")
    x_poisson = []
    x_poisson_only = []
    snr = 10 ** (snr / 10.0)
    for i in range(x.shape[0]):
        signal = np.array(x[i])
        signal = np.squeeze(signal)
        xpower = np.sum(signal ** 2) / len(signal)
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
    # x_gussian_only = np.expand_dims(x_gussian_only, 1)

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


# 将labels中normal_class变为0，其他变为1
def one_class_labeling(labels, normal_class: int, seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)  # 洗牌
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


class GussianNoisyDataset(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """

        X_gussian, _ = Gussian_Noisy(X, SNR)
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
        # X原始信号 Y为标签
        # 噪声，信号与噪声混合
        # 导联x1,x2
        # x1,x2=np.split(X,[1],axis=1)       #(704,2,1600)----------->(704,1,1600), (704,1,1600)
        # x1=np.squeeze(x1,axis=1)
        # x2=np.squeeze(x2,axis=1)
        Noisy, Gussain, _ = Gussian_Noisy_return_N(X, SNR)
        # X = np.expand_dims(X,1)
        self.X = torch.Tensor(X)
        self.fix = Gussain
        self.Nosiy_Only = torch.Tensor(Noisy)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.fix[index], self.Nosiy_Only[index], self.Y[index]

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


class GammaNoisyDataset_return_N(data.Dataset):
    def __init__(self, X, Y, SNR):
        """
        """
        Noisy, Gamma, _ = Gamma_Noisy_return_N(X, SNR)
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
# 信号随机加噪声 单导联
class fix_one_data(data.Dataset):
    def __init__(self, X, Y, SNR):  # 原始信号(704,2,1600)，标签，信噪比
        # 噪声标签
        # 0:Gussian
        # 1:Possion
        # 2:Uniform
        # 3:Exponential
        # 4:Rayleign
        # 5:Gamma
        fix_list = []
        list_nosiy_label = [x % 6 for x in range(X.shape[0])]
        list_nosiy_data = []
        np.random.shuffle(list_nosiy_label)
        X=np.expand_dims(X,1)
        for i in range(X.shape[0]):
            if list_nosiy_label[i] == 0:
                Noisy, fix, _ = Gussian_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 1:
                Noisy, fix, _ = Poisson_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 2:
                Noisy, fix, _ = Uniform_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 3:
                Noisy, fix, _ = Exponential_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 4:
                Noisy, fix, _ = Rayleign_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 5:
                Noisy, fix, _ = Gamma_Noisy_return_N(X[i], SNR)
            fix_list.append(fix)
            list_nosiy_data.append(Noisy)
        self.X = torch.Tensor(X)
        #Fix=[np.expand_dims(i,1) for i in fix_list]
        self.fix = torch.Tensor(fix_list)
        # 标签
        self.Y = torch.Tensor(Y)
        #N_data=[np.expand_dims(i,1) for i in list_nosiy_data]
        self.Nosiy_Only = torch.Tensor(list_nosiy_data)
        # 噪声标签
        self.Nosiy_label = torch.Tensor(list_nosiy_label)
    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.fix[index], self.Y[index], self.Nosiy_Only[index], self.Nosiy_label[index]

    def __len__(self):
        return self.X.size(0)
# 信号随机加噪声 多导联
class fix_data(data.Dataset):
    def __init__(self, X, Y, SNR):  # 原始信号(704,2,1600)，标签，信噪比
        # 噪声标签
        # 0:Gussian
        # 1:Possion
        # 2:Uniform
        # 3:Exponential
        # 4:Rayleign
        # 5:Gamma
        fix_list = []
        list_nosiy_label = [x % 6 for x in range(X.shape[0])]
        list_nosiy_data = []
        np.random.shuffle(list_nosiy_label)
        for i in range(X.shape[0]):
            if list_nosiy_label[i] == 0:
                Noisy, fix, _ = Gussian_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 1:
                Noisy, fix, _ = Poisson_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 2:
                Noisy, fix, _ = Uniform_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 3:
                Noisy, fix, _ = Exponential_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 4:
                Noisy, fix, _ = Rayleign_Noisy_return_N(X[i], SNR)
            elif list_nosiy_label[i] == 5:
                Noisy, fix, _ = Gamma_Noisy_return_N(X[i], SNR)
            fix_list.append(fix)
            list_nosiy_data.append(Noisy)
        self.X = torch.Tensor(X)
        self.fix = torch.Tensor(fix_list)
        self.Y = torch.Tensor(Y)
        self.Nosiy_Only = torch.Tensor(list_nosiy_data)
        self.Nosiy_label = torch.Tensor(list_nosiy_label)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.fix[index], self.Y[index], self.Nosiy_Only[index], self.Nosiy_label[index]

    def __len__(self):
        return self.X.size(0)


def WavePlot_Single(x1, name):
    x = np.linspace(0, len(x1), len(x1))
    a = list(x1)

    c = np.array(a)
    y = c

    plt.plot(x, y, ls="-", color="b", marker=",", lw=2)
    plt.axis('on')
    # plt.legend()

    # plt.show()
    plt.savefig('{}.svg'.format(name))
    plt.close()


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


def load_data(opt, dataset_name):
    if dataset_name in ['CSPC']:
        data_X = np.load("/home/huangxunhua/ECG_platform/ECG/experiments/datasets/CSPC_ECG/cpsc_data.npy")
        data_Y = np.load("/home/huangxunhua/ECG_platform/ECG/experiments/datasets/CSPC_ECG/cpsc_label.npy")
        # data_X = np.load("/home/chenpeng/workspace/Noisy_MultiModal/experiments/datasets/CSPC/cpsc_data.npy")
        # data_Y = np.load("/home/chenpeng/workspace/Noisy_MultiModal/experiments/datasets/CSPC/cpsc_label.npy")
        # data_nomal=np.load("/home/chenpeng/workspace/dataset/CSPC2021_fanc/cpsc_nomal.npy")
        # data_abnomal=np.load("/home/chenpeng/workspace/dataset/CSPC2021_fanc/cpsc_fangchan.npy")

        # len_nomal=data_nomal.shape[0]
        # len_abnomal=data_abnomal.shape[0]
        # train_data=data_nomal[:int(0.6*len_nomal),]
        # train_label=np.zeros(train_data.shape[0])
        # val_data_nomal=data_nomal[int(0.6*len_nomal):int(0.8*len_nomal),]
        # val_data_abnomal=data_abnomal[:int(0.5*len_abnomal),]
        # val_data=np.concatenate([val_data_nomal,val_data_abnomal],axis=0)
        # val_label=np.concatenate([np.zeros(val_data_nomal.shape[0]),np.ones(val_data_abnomal.shape[0])])
        # test_data_nomal=data_nomal[int(0.8*len_nomal):,]
        # test_data_abnomal=data_abnomal[int(0.5*len_abnomal):,]
        # test_data = np.concatenate([test_data_nomal,test_data_abnomal], axis=0)
        # test_label=np.concatenate([np.zeros(test_data_nomal.shape[0]),np.ones(test_data_abnomal.shape[0])])
        #
        #
        # # 打乱顺序
        # labels_binary, idx_normal, idx_abnormal = one_class_labeling(train_label,0,opt.seed)


        print()



    elif dataset_name in ['MFPT']:
        data_X = np.load("{}/{}/{}_data.npy".format(opt.data_UCR, dataset_name, dataset_name))
        # (2574,1024)
        data_Y = np.load("{}/{}/{}_label.npy".format(opt.data_UCR, dataset_name, dataset_name))

    elif dataset_name in ['CWRU']:
        with open('{}/{}/{}_data.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle1:
            data_X = pickle.load(handle1)
            # (8768,1024)

        with open('{}/{}/{}_label.pickle'.format(opt.data_UCR, dataset_name, dataset_name), 'rb') as handle2:
            data_Y = pickle.load(handle2)

    elif dataset_name in ['EpilepticSeizure']:

        data_X, data_Y = get_EpilepticSeizure(opt.data_UCR, dataset_name)


    else:

        train_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TRAIN.tsv')),
                                delimiter='\t')  #
        test_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TEST.tsv')),
                               delimiter='\t')  #

        data_ALL = np.concatenate((train_data, test_data), axis=0)
        data_X = data_ALL[:, 1:]  # (16637,96)
        data_Y = data_ALL[:, 0] - min(data_ALL[:, 0])  # (16637,)

    # data_X = rescale(data_X)

    # data_X, data_Y = get_EpilepticSeizure(opt.data_UCR, dataset_name)

    label_idxs = np.unique(data_Y)
    # 统计各类别个数
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(data_Y == idx)[0])

    if opt.normal_idx >= len(label_idxs):
        normal_idx = opt.normal_idx % len(label_idxs)
    else:
        normal_idx = opt.normal_idx

    if dataset_name in ['EpilepticSeizure']:
        labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)
    else:
        labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)

    data_N_X = data_X[idx_normal]  # (732,2,1600)
    data_N_Y = labels_binary[idx_normal]  # (732,)  1D
    data_A_X = data_X[idx_abnormal]  # (704,2,1600)
    data_A_Y = labels_binary[idx_abnormal]  # (704,)  1D

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

    # if opt.normalize:  # 归一化
    #     print("[INFO] Data Normalization!")
    #     # Normalize
    #     x_train_max = np.max(train_X)
    #     x_train_min = np.min(train_X)
    #     train_X = 2. * (train_X - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
    #     # Test is secret
    #     val_X = 2. * (val_X - x_train_min) / (x_train_max - x_train_min) - 1.
    #     test_X = 2. * (test_X - x_train_min) / (x_train_max - x_train_min) - 1.

    print("[INFO] Labels={}, normal label={}".format(label_idxs, opt.normal_idx))
    print("[INFO] Train: normal={}".format(train_X.shape), )
    print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
    print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

    # Wavelet transform
    X_length = train_X.shape[-1]

    # transform = transforms.ToTensor()

    signal_length = [0]

    if opt.model in ['AE_CNN_noisy_multi']:
        print(opt.Snr)

        if opt.NoisyType == 'Gussian':

            _, _, signal_length = Gussian_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GussianNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = GussianNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = GussianNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Rayleign':
            _, _, signal_length = Rayleign_Noisy_return_N(train_X, opt.Snr)
            train_dataset = RayleignNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = RayleignNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = RayleignNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Exponential':
            _, _, signal_length = Exponential_Noisy_return_N(train_X, opt.Snr)
            train_dataset = ExponentialNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = ExponentialNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = ExponentialNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Uniform':
            _, _, signal_length = Uniform_Noisy_return_N(train_X, opt.Snr)
            train_dataset = UniformNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = UniformNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = UniformNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType == 'Poisson':
            _, _, signal_length = Poisson_Noisy_return_N(train_X, opt.Snr)
            train_dataset = PossionNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = PossionNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = PossionNoisyDataset_return_N(test_X, test_Y, opt.Snr)

        elif opt.NoisyType =='Gamma':
            _, _, signal_length = Gamma_Noisy_return_N(train_X, opt.Snr)
            train_dataset = GammaNoisyDataset_return_N(train_X, train_Y, opt.Snr)
            val_dataset = GammaNoisyDataset_return_N(val_X, val_Y, opt.Snr)
            test_dataset = GammaNoisyDataset_return_N(test_X, test_Y, opt.Snr)
        else:
            print("illegal noisy type")

    elif opt.model in [ 'AE_CNN_self','AE_CNN_self_2']:
        if opt.nc==1:
            train_dataset = fix_one_data(train_X, train_Y, opt.Snr)
            val_dataset = fix_one_data(val_X, val_Y, opt.Snr)
            test_dataset = fix_one_data(test_X, test_Y, opt.Snr)
        else:
            train_dataset = fix_data(train_X, train_Y, opt.Snr)
            val_dataset = fix_data(val_X, val_Y, opt.Snr)
            test_dataset = fix_data(test_X, test_Y, opt.Snr)

    else:
        # train_X = np.expand_dims(train_X, 1)  # (292,1,140)
        # test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
        # val_X = np.expand_dims(val_X, 1)
        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

    dataloader = {"train": DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=opt.batchsize,  # mini batch size
        shuffle=True,
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
            drop_last=True)
    }

    return dataloader, X_length, signal_length
