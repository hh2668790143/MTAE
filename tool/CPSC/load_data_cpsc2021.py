# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/11/19 20:18
import copy
import glob
import pickle
import random
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
import numpy as np


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

    # snr=350
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

    # snr=300
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

    # snr=250
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
    # snr=100
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


class fix_data(data.Dataset):
    def __init__(self, X, Y, seed, SNR):  # 原始信号(704,2,1600)，标签，信噪比
        # 噪声标签
        # 0:Gussian
        # 1:Possion
        # 2:Uniform
        # 3:Exponential
        # 4:Rayleign
        # 5:Gamma
        global fix, Noisy
        fix_list = []
        list_nosiy_label = [x % 5 for x in range(X.shape[0])]
        list_nosiy_data = []
        np.random.seed(seed)
        np.random.shuffle(list_nosiy_label)
        for i in range(X.shape[0]):
            if list_nosiy_label[i] == 0:
                Noisy, fix, _ = Gussian_Noisy_return_N(X[i], SNR[0])
            elif list_nosiy_label[i] == 1:
                Noisy, fix, _ = Uniform_Noisy_return_N(X[i], SNR[1])
            elif list_nosiy_label[i] == 2:
                Noisy, fix, _ = Exponential_Noisy_return_N(X[i], SNR[2])
            elif list_nosiy_label[i] == 3:
                Noisy, fix, _ = Rayleign_Noisy_return_N(X[i], SNR[3])
            elif list_nosiy_label[i] == 4:
                Noisy, fix, _ = Gamma_Noisy_return_N(X[i], SNR[4])
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


def read_pickle(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            res = pickle.load(f)
    return res


def add_normal_patient(normal, root):
    all_normal_patient = []
    for patient_id in normal:
        normal_patient = []
        for i in range(30):
            res = read_pickle(root + "data_{}_{}_normal.pickle".format(patient_id, i))
            if res is not None:
                normal_patient.append(res)
        all_normal_patient.append(normal_patient)
    return all_normal_patient


def add_abnormal_patient(normal, root):
    all_normal_patient = []
    for patient_id in normal:
        normal_patient = []
        for i in range(30):
            res = read_pickle(root + "data_{}_{}_fangchan.pickle".format(patient_id, i))
            if res is not None:
                normal_patient.append(res)
        all_normal_patient.append(normal_patient)
    return all_normal_patient


def add_normal_abnormal_patient(normal, root):
    return add_normal_patient(normal, root), add_abnormal_patient(normal, root)


def List_2_Numpy(list):
    z_list = []  # 每个元素代表一个病人
    for i in list:
        z_list.append(np.concatenate(i, axis=0))
    return np.array(z_list, dtype=object)


def len_data(list):
    return sum([i.shape[0] for i in list])


def get_data(root):
    normal_path = glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_normal.pickle")
    fangchan_path = glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_fangchan.pickle")
    # 获得患者id
    normal_patient = sorted([int(x.split("/")[-1].split("_")[1]) for x in normal_path])
    fangchan_patient = sorted([int(x.split("/")[-1].split("_")[1]) for x in fangchan_path])

    normal_patient = list(set(normal_patient))
    fangchan_patient = list(set(fangchan_patient))

    normal_abnormal = np.intersect1d(normal_patient, fangchan_patient)  # 23

    normal = np.array([i for i in normal_patient if i not in normal_abnormal])  # 51
    abnormal = np.array([i for i in fangchan_patient if i not in normal_abnormal])  # 31

    # 非房颤
    all_normal_patient = add_normal_patient(normal, root)
    # 房颤
    all_abnormal_patient = add_abnormal_patient(abnormal, root)
    # 既有房颤又有非房颤
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = add_normal_abnormal_patient(normal_abnormal, root)

    all_normal_patient = List_2_Numpy(all_normal_patient)  # (51,) 96600
    all_abnormal_patient = List_2_Numpy(all_abnormal_patient)  # (31,) 52682
    all_normal_abnormal_patient_N = List_2_Numpy(all_normal_abnormal_patient_N)  # (23,) 8477
    all_normal_abnormal_patient_A = List_2_Numpy(all_normal_abnormal_patient_A)  # (23,) 6105
    # 163864

    return all_normal_patient, all_abnormal_patient, all_normal_abnormal_patient_N, all_normal_abnormal_patient_A


def get_data_len(data, lens):
    train_data = []
    for i in range(50000):
        for j in range(data.shape[0]):
            if i < data[0].shape[0]:
                train_data.append(np.expand_dims(data[0][i], axis=0))
            if len(train_data) == lens:
                break
        if len(train_data) == lens:
            break

    return np.concatenate(train_data, axis=0)


def load_data_1(root):
    all_normal_patient, all_abnormal_patient, all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(
        root)
    # 51 normal
    # 23 abnormal

    train_normal_data = all_normal_patient[:30, ]  # 58932
    else_normal_data = all_normal_patient[30:, ]  # 37668

    test_normal_data = np.concatenate([else_normal_data, all_normal_abnormal_patient_N], axis=0)  # 46145
    test_abnormal_data = np.concatenate([all_abnormal_patient, all_normal_abnormal_patient_A], axis=0)  # 58787

    train_normal_data = get_data_len(train_normal_data, 5000)
    test_normal_data = get_data_len(test_normal_data, 5000)
    test_abnormal_data = get_data_len(test_abnormal_data, 5000)

    # test_normal_data = np.concatenate(test_normal_data, axis=0)
    # test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)

    # np.save(save + "/train_normal_data.npy", train_normal_data)
    # np.save(save + "/test_normal_data.npy", test_normal_data)
    # np.save(save + "/test_abnormal_data.npy", test_abnormal_data)
    return train_normal_data, test_normal_data, test_abnormal_data


def load_data(root, len_num=1000, seed=1024):
    all_normal_patient, all_abnormal_patient, \
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []

    for i in all_normal_patient[:40]:  # 41
        np.random.seed(seed)
        np.random.shuffle(i)
        train_data.append(i[:600, :, :len_num])
    test_normal_data = []
    test_abnormal_data = []
    val_normal_data = []
    val_abnormal_data = []
    test_all_normal_patient = all_normal_patient[40:]

    np.random.seed(seed)
    np.random.shuffle(all_abnormal_patient)
    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_A)
    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_N)
    np.random.seed(seed)
    np.random.shuffle(test_all_normal_patient)
    # val
    for i in all_abnormal_patient[:15]:  # 31
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[:11]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in test_all_normal_patient[:5]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:400, :, :len_num])  # 400
    # k = 0
    for i in all_normal_abnormal_patient_N[:11]:  # 23
    #     k += 1
    #     print(k)
    #     for j in range(10):
    #         plot_sample_2(i[j], i[j + 10],
    #                       save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/tool/img',
    #                       datename="Normal_num" + str(k) + "_" + str(j))
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:100, :, :len_num])
    # test
    for i in all_abnormal_patient[15:]:  # 31
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[11:]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in test_all_normal_patient[5:]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:400, :, :len_num])  # 400

    for i in all_normal_abnormal_patient_N[11:]:  # 23
        # k += 1
        # print(k)
        # for j in range(10):
        #     plot_sample_2(i[j], i[j + 10],
        #                   save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/tool/img',
        #                   datename="Normal_num" + str(k) + "_" + str(j))
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:100, :, :len_num])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    val_normal_data = np.concatenate(val_normal_data, axis=0)
    val_abnormal_data = np.concatenate(val_abnormal_data, axis=0)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/train_data.npy", train_data)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/test_normal_data.npy", test_normal_data)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/test_abnormal_data.npy", test_abnormal_data)
    return train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data


def load_data_ant(root, len_num=1000):
    all_normal_patient, all_abnormal_patient, \
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []
    test_normal_data = []
    test_abnormal_data = []

    # for i in all_normal_patient: #51
    #     test_normal_data.append(i[:120, : ,:len_num])

    for i in all_abnormal_patient:  # 31
        train_data.append(i[:100, :, :len_num])

    for i in all_normal_abnormal_patient_A:  # 23
        test_normal_data.append(i[:50, :, :len_num])

    for i in all_normal_abnormal_patient_N:  # 23
        test_abnormal_data.append(i[:50, :, :len_num])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/train_data.npy", train_data)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/test_normal_data.npy", test_normal_data)
    # np.save("/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/test_abnormal_data.npy", test_abnormal_data)
    return train_data, test_normal_data, test_abnormal_data


def load_data_2(root):
    all_normal_patient, all_abnormal_patient, \
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []
    for i in all_normal_patient[:50]:
        train_data.append(i[:100, ])  # 2000条
    test_normal_data = []
    test_abnormal_data = []

    for i in all_normal_abnormal_patient_N:
        test_abnormal_data.append(i[:30, ])

    for i in all_normal_abnormal_patient_A:
        test_normal_data.append(i[:50, ])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)

    return train_data, test_normal_data, test_abnormal_data


def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index


def load_CPSC2021_data(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED):
    # train_normal_data, test_normal_data, test_abnormal_data = load_data(root)

    len_train = train_data.shape[0]
    len_val_normal = val_normal_data.shape[0]
    len_val_abnormal = val_abnormal_data.shape[0]

    len_test_normal = test_normal_data.shape[0]
    len_test_abnormal = test_abnormal_data.shape[0]

    train_label = np.zeros(len_train)
    val_data = np.concatenate([val_normal_data, val_abnormal_data], axis=0)
    val_label = np.concatenate([np.zeros(len_val_normal), np.ones(len_val_abnormal)], axis=0)

    test_data = np.concatenate([test_normal_data, test_abnormal_data], axis=0)
    test_label = np.concatenate([np.zeros(len_test_normal), np.ones(len_test_abnormal)], axis=0)

    train_label, train_idx = shuffle_label(train_label, SEED)
    train_data = train_data[train_idx]

    val_label, val_idx = shuffle_label(val_label, SEED)
    val_data = val_data[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_label, SEED)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]

    return train_data, train_label, val_data, val_label, test_data, test_label


def load_d(root):
    all_normal_patient, all_abnormal_patient, \
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []
    for i in all_abnormal_patient[:30]:
        train_data.append(i[:100, ])

    test_normal_data = []
    test_abnormal_data = []

    for i in all_normal_abnormal_patient_N[:10]:
        test_abnormal_data.append(i[:100, ])

    for i in all_normal_abnormal_patient_A[:10]:
        test_normal_data.append(i[:100, ])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)

    return train_data, test_normal_data, test_abnormal_data


def getDataSet(train_data, train_label, test_data, test_label, opt, K=5, idx=4):  # K份验证 idx:第几分验证
    split_num = int(train_data.shape[0] / K)
    global train_X, test_X
    if opt.normalize:  # 归一化
        print("[INFO] Data Normalization!")
        # Normalize
        x_train_max = np.max(train_data)
        x_train_min = np.min(train_data)
        train_X = 2. * (train_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
        # Test is secret
        test_X = 2. * (test_data - x_train_min) / (x_train_max - x_train_min) - 1.
    split_train = []
    for i in range(K):
        split_train.append(train_X[i * split_num:(i + 1) * split_num, ])
    val_X = train_X[idx * split_num:(idx + 1) * split_num, ]
    train_X = np.concatenate([train_X[:idx * split_num, ], train_X[(idx + 1) * split_num:, ]], axis=0)
    val_label = train_label[idx * split_num:(idx + 1) * split_num, ]
    train_Y = np.concatenate([train_label[:idx * split_num, ], train_label[(idx + 1) * split_num:, ]], axis=0)

    if opt.model in ['AE_CNN_self']:
        train_dataset = fix_data(train_X, train_Y, opt.seed, opt.Snr)
        val_dataset = fix_data(val_X, val_label, opt.seed, opt.Snr)
        test_dataset = fix_data(test_X, test_label, opt.seed, opt.Snr)
    else:
        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_label)
        test_dataset = RawDataset(test_X, test_label)

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

    return dataloader, train_X.shape[-1]


# def data_2_dataset_2(root, opt):
#     train_data, train_label, test_data, test_label = load_CPSC2021_data_1(root, opt.seed)
#     return getDataSet(train_data, train_label, test_data, test_label, opt)


def data_2_dataset(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, opt):
    global train_X, val_X, test_X
    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)
    if opt.normalize:  # 归一化
        print("[INFO] Data Normalization!")
        # Normalize
        x_train_max = np.max(train_data)
        x_train_min = np.min(train_data)
        train_X = 2. * (train_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
        # Test is secret
        val_X = 2. * (val_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
        test_X = 2. * (test_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
    # train_X=np.concatenate([train_X,train_X[::-1]],axis=0)
    # train_X1=copy.deepcopy(train_X)
    # for i in range(train_X1.shape[0]):
    #     train_X1[i][0]=train_X1[i][0][::-1]
    #     train_X1[i][1]=train_X1[i][1][::-1]
    # train_X = np.concatenate([train_X, train_X1], axis=0)
    # train_label=np.concatenate([train_label,train_label[::-1]],axis=0)
    if opt.model in ['AE_CNN_self', 'AE_CNN_self_2', 'AE_CNN_self_3']:
        train_dataset = fix_data(train_X, train_label, opt.seed, opt.Snr)
        val_dataset = fix_data(val_X, val_label, opt.seed, opt.Snr)
        test_dataset = fix_data(test_X, test_label, opt.seed, opt.Snr)
    else:
        train_dataset = RawDataset(train_X, train_label)
        val_dataset = RawDataset(val_X, val_label)
        test_dataset = RawDataset(test_X, test_label)

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
            dataset=test_dataset,  # torch TensorDataset format val_dataset
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True)
    }

    return dataloader, train_X.shape[-1]


if __name__ == '__main__':
    from MlultModal.options import Options
    from MlultModal.draw_ECG.draw_ecg import plot_sample_2

    root = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/"
    save = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/"
    load_data(root, 1000)
    print("ok")
    # load_CPSC2021_data_1(root, 4)
    # opt = Options().parse()
    # opt.model="AE_CNN_self"
    # opt.Snr=20
    # opt.seed=4
    # dataloader, lens=data_2_dataset(root,opt)
    # for datas in dataloader["train"]:
    #     signal=datas[0]
    #     fix_signal=datas[1]
    #     label=datas[4]
    #     for i in range(signal.shape[0]):
    #         plot_sample_2(signal=signal[i].numpy(),
    #                       signal_noise=fix_signal[i].numpy(),
    #                       # rec_signal=self.rec_singal[i].cpu.numpy(),
    #                       noise_label=int(label[i].numpy()),
    #                       save_dir='/home/chenpeng/workspace/Noisy_MultiModal/experiments/MlultModal/img',
    #                       datename=str(i))
