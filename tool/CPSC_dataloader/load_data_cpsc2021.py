import copy
import glob
import pickle
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
import numpy as np
import sys

from tool.CPSC_dataloader import transform_f

sys.path.append('/home/chenpeng/workspace/Noisy_MultiModal/experiments/')
from tool.CPSC_dataloader import transform
from tool.CPSC_dataloader.get_noise import *


class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """
        # X = np.expand_dims(X, 1)
        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class RawDataset2(data.Dataset):
    def __init__(self, X_J, X, Y):
        """
        """

        self.X = torch.Tensor(X)
        self.X_J = torch.Tensor(X_J)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index], self.X_J[index]

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
        np.random.seed(seed)
        list_nosiy_label = np.random.randint(0, 5, X.shape[0])
        list_nosiy_data = []

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


class TransformDataset(data.Dataset):
    def __init__(self, X, Y, opt, snr=5, train=1, p=0.5):
        if opt.FM:
            fix_data = []  # 混合噪声
            fix_noisy_data = []  # 噪声混合
            labels = []
            fix_data_f = []  # 混合噪声
            fix_noisy_data_f = []  # 噪声混合
            labels_f = []
            # X = np.expand_dims(X, 1)
            for i in X:
                gussian = transform.Gussian(snr=snr, p=p)
                gamma = transform.Gamma(snr=snr, p=p)
                rayleign = transform.Rayleign(snr=snr, p=p)
                exponential = transform.Exponential(snr=snr, p=p)
                # poisson = transform.Poisson(snr=snr, p=p)
                uniform = transform.Uniform(snr=snr, p=p)
                transforms_list = {
                    'gussian': gussian,
                    'gamma': gamma,
                    'rayleign': rayleign,
                    'exponential': exponential,
                    # 'poisson': poisson,
                    'uniform': uniform
                }
                trans = transform.Compose(transforms_list.values(), snr=snr)
                data, fix_noisy, label = trans(i)
                labels.append(self.list2hot(label))
                fix_data.append(data)
                fix_noisy_data.append(fix_noisy)
                # ============================================== filter
                kalman = transform_f.Kalman(p=p)
                wiener = transform_f.Wiener(p=p)
                transforms_list_f = {
                    'kalman': kalman,
                    # 'wiener': wiener,
                }
                trans_f = transform_f.Compose(transforms_list_f.values())
                data_f, fix_noisy_f, label_f = trans_f(i)
                labels_f.append(self.list2hot_f(label_f))
                fix_data_f.append(data_f)
                fix_noisy_data_f.append(fix_noisy_f)

            fix_data = np.array(fix_data)
            fix_noisy_data = np.array(fix_noisy_data)
            labels = np.array(labels)
            fix_data_f = np.array(fix_data_f)
            fix_noisy_data_f = np.array(fix_noisy_data_f)
            labels_f = np.array(labels_f)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f
            with open("./data{}_pkl/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed)), 'wb') as f:
                pickle.dump(data, f)
            # np.save("{}_{}_{}".format(opt.dataset, str(opt.Snr), str(opt.seed)), data)
        else:
            # data = np.load("{}_{}_{}".format(opt.dataset, str(opt.Snr), str(opt.seed)) + ".npy", allow_pickle=True)

            with open("./data{}_pkl/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed)), 'rb') as f:
                data = pickle.load(f)

            fix_data = data['fix_data']
            fix_noisy_data = data['fix_noisy_data']
            labels = data['labels']
            fix_data_f = data['fix_data_f']
            fix_noisy_data_f = data['fix_noisy_data_f']
            labels_f = data['labels_f']

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)

    def list2hot(self, label):
        res = []
        # for i in ['Gussian','Gamma','Rayleign','Exponential','Poisson','Uniform']:
        for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def list2hot_f(self, label):
        res = []
        for i in ['Kalman', 'Wiener']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]

    def __len__(self):
        return self.X.size(0)


class TransformDataset1(data.Dataset):
    def __init__(self, X, Y, snr=5, p=0.5):
        fix_data = []  # 混合噪声
        fix_noisy_data = []  # 噪声混合
        labels = []
        # X = np.expand_dims(X, 1)
        for i in X:
            gussian = transform.Gussian(snr=snr, p=p)
            gamma = transform.Gamma(snr=snr, p=p)
            rayleign = transform.Rayleign(snr=snr, p=p)
            exponential = transform.Exponential(snr=snr, p=p)
            # poisson = transform.Poisson(snr=snr, p=p)
            uniform = transform.Uniform(snr=snr, p=p)
            transforms_list = {
                'gussian': gussian,
                'gamma': gamma,
                'rayleign': rayleign,
                'exponential': exponential,
                # 'poisson': poisson,
                'uniform': uniform
            }
            trans = transform.Compose(transforms_list.values(), snr=snr)
            data, fix_noisy, label = trans(i)
            labels.append(self.list2hot(label))
            fix_data.append(data)
            fix_noisy_data.append(fix_noisy)

        fix_data = np.array(fix_data)
        fix_noisy_data = np.array(fix_noisy_data)
        labels = np.array(labels)

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

    def list2hot(self, label):
        res = []
        # for i in ['Gussian','Gamma','Rayleign','Exponential','Poisson','Uniform']:
        for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def list2hot_f(self, label):
        res = []
        for i in ['Kalman']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index]

    def __len__(self):
        return self.X.size(0)


class TransformDataset2(data.Dataset):
    def __init__(self, X, Y, opt, snr=5, train=1, p=0.5):
        if opt.FM:
            fix_data = []  # 混合噪声
            fix_noisy_data = []  # 噪声混合
            labels = []
            fix_data_f = []  # 混合噪声
            fix_noisy_data_f = []  # 噪声混合
            labels_f = []
            # X = np.expand_dims(X, 1)
            for i in X:
                gussian = transform.Gussian(snr=snr, p=p)
                gamma = transform.Gamma(snr=snr, p=p)
                rayleign = transform.Rayleign(snr=snr, p=p)
                exponential = transform.Exponential(snr=snr, p=p)
                # poisson = transform.Poisson(snr=snr, p=p)
                uniform = transform.Uniform(snr=snr, p=p)
                transforms_list = {
                    'gussian': gussian,
                    'gamma': gamma,
                    'rayleign': rayleign,
                    'exponential': exponential,
                    # 'poisson': poisson,
                    'uniform': uniform
                }
                trans = transform.Compose(transforms_list.values(), snr=snr)
                data, fix_noisy, label = trans(i)
                labels.append(self.list2hot(label))
                fix_data.append(data)
                fix_noisy_data.append(fix_noisy)
                # ============================================== filter
                kalman = transform_f.Kalman(p=p)
                wiener = transform_f.Wiener(p=p)
                transforms_list_f = {
                    'kalman': kalman,
                    # 'wiener': wiener,
                }
                trans_f = transform_f.Compose(transforms_list_f.values())
                data_f, fix_noisy_f, label_f = trans_f(i)
                labels_f.append(self.list2hot_f(label_f))
                fix_data_f.append(data_f)
                fix_noisy_data_f.append(fix_noisy_f)

            fix_data = np.array(fix_data)
            fix_noisy_data = np.array(fix_noisy_data)
            labels = np.array(labels)
            fix_data_f = np.array(fix_data_f)
            fix_noisy_data_f = np.array(fix_noisy_data_f)
            labels_f = np.array(labels_f)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f

            with open("./data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                            str(opt.normal_idx)), 'wb') as f:
                pickle.dump(data, f)

        else:

            with open("./data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                            str(opt.normal_idx)), 'rb') as f:
                data = pickle.load(f)

            fix_data = data['fix_data']
            fix_noisy_data = data['fix_noisy_data']
            labels = data['labels']
            fix_data_f = data['fix_data_f']
            fix_noisy_data_f = data['fix_noisy_data_f']
            labels_f = data['labels_f']

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)

    def list2hot(self, label):
        res = []
        # for i in ['Gussian','Gamma','Rayleign','Exponential','Poisson','Uniform']:
        for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def list2hot_f(self, label):
        res = []
        for i in ['Kalman', 'Wiener']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]

    def __len__(self):
        return self.X.size(0)


class TransformDataset_l1(data.Dataset):
    def __init__(self, X, Y, opt, snr=5, train=1, p=0.5):
        if opt.FM:
            fix_data = []  # 混合噪声
            fix_noisy_data = []  # 噪声混合
            labels = []
            fix_data_f = []  # 混合噪声
            fix_noisy_data_f = []  # 噪声混合
            labels_f = []
            # X = np.expand_dims(X, 1)
            for i in X:
                gussian = transform.Gussian(snr=snr, p=p)
                gamma = transform.Gamma(snr=snr, p=p)
                rayleign = transform.Rayleign(snr=snr, p=p)
                exponential = transform.Exponential(snr=snr, p=p)
                # poisson = transform.Poisson(snr=snr, p=p)
                uniform = transform.Uniform(snr=snr, p=p)
                transforms_list = {
                    'gussian': gussian,
                    'gamma': gamma,
                    'rayleign': rayleign,
                    'exponential': exponential,
                    # 'poisson': poisson,
                    'uniform': uniform
                }
                trans = transform.Compose(transforms_list.values(), snr=snr)
                data, fix_noisy, label = trans(i)
                labels.append(self.list2hot(label))
                fix_data.append(data)
                fix_noisy_data.append(fix_noisy)
                # ============================================== filter
                l1 = transform_f.L1_filter(p=p)
                transforms_list_f = {
                    'l1': l1,
                }
                trans_f = transform_f.Compose(transforms_list_f.values())
                data_f, fix_noisy_f, label_f = trans_f(i)
                labels_f.append(self.list2hot_f(label_f))
                fix_data_f.append(data_f)
                fix_noisy_data_f.append(fix_noisy_f)

            fix_data = np.array(fix_data)
            fix_noisy_data = np.array(fix_noisy_data)
            labels = np.array(labels)
            fix_data_f = np.array(fix_data_f)
            fix_noisy_data_f = np.array(fix_noisy_data_f)
            labels_f = np.array(labels_f)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f

            if not os.path.exists(
                    "./l1_data{}_pkl/{}".format(train, opt.dataset)):
                os.makedirs("./l1_data{}_pkl/{}".format(train, opt.dataset))

            with open("./l1_data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                               str(opt.normal_idx)), 'wb') as f:
                pickle.dump(data, f)

        else:

            with open("./l1_data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                               str(opt.normal_idx)), 'rb') as f:
                data = pickle.load(f)

            fix_data = data['fix_data']
            fix_noisy_data = data['fix_noisy_data']
            labels = data['labels']
            fix_data_f = data['fix_data_f']
            fix_noisy_data_f = data['fix_noisy_data_f']
            labels_f = data['labels_f']

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)

    def list2hot(self, label):
        res = []
        # for i in ['Gussian','Gamma','Rayleign','Exponential','Poisson','Uniform']:
        for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def list2hot_f(self, label):
        res = []
        for i in ['Kalman', 'Wiener']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]

    def __len__(self):
        return self.X.size(0)


class TransformDataset_hp(data.Dataset):
    def __init__(self, X, Y, opt, snr=5, train=1, p=0.5):
        if opt.FM:
            fix_data = []  # 混合噪声
            fix_noisy_data = []  # 噪声混合
            labels = []
            fix_data_f = []  # 混合噪声
            fix_noisy_data_f = []  # 噪声混合
            labels_f = []
            # X = np.expand_dims(X, 1)
            for i in X:
                gussian = transform.Gussian(snr=snr, p=p)
                gamma = transform.Gamma(snr=snr, p=p)
                rayleign = transform.Rayleign(snr=snr, p=p)
                exponential = transform.Exponential(snr=snr, p=p)
                # poisson = transform.Poisson(snr=snr, p=p)
                uniform = transform.Uniform(snr=snr, p=p)
                transforms_list = {
                    'gussian': gussian,
                    'gamma': gamma,
                    'rayleign': rayleign,
                    'exponential': exponential,
                    # 'poisson': poisson,
                    'uniform': uniform
                }
                trans = transform.Compose(transforms_list.values(), snr=snr)
                data, fix_noisy, label = trans(i)
                labels.append(self.list2hot(label))
                fix_data.append(data)
                fix_noisy_data.append(fix_noisy)
                # ============================================== filter
                hp = transform_f.Hp_filter(p=p)
                transforms_list_f = {
                    'hp': hp,
                }
                trans_f = transform_f.Compose(transforms_list_f.values())
                data_f, fix_noisy_f, label_f = trans_f(i)
                labels_f.append(self.list2hot_f(label_f))
                fix_data_f.append(data_f)
                fix_noisy_data_f.append(fix_noisy_f)

            fix_data = np.array(fix_data)
            fix_noisy_data = np.array(fix_noisy_data)
            labels = np.array(labels)
            fix_data_f = np.array(fix_data_f)
            fix_noisy_data_f = np.array(fix_noisy_data_f)
            labels_f = np.array(labels_f)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f

            if not os.path.exists(
                    "./hp_data{}_pkl/{}".format(train, opt.dataset)):
                os.makedirs("./hp_data{}_pkl/{}".format(train, opt.dataset))

            with open("./hp_data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                               str(opt.normal_idx)), 'wb') as f:
                pickle.dump(data, f)

        else:

            with open("./hp_data{}_pkl/{}/{}_{}_{}.pkl".format(train, opt.dataset, str(opt.Snr), str(opt.seed),
                                                               str(opt.normal_idx)), 'rb') as f:
                data = pickle.load(f)

            fix_data = data['fix_data']
            fix_noisy_data = data['fix_noisy_data']
            labels = data['labels']
            fix_data_f = data['fix_data_f']
            fix_noisy_data_f = data['fix_noisy_data_f']
            labels_f = data['labels_f']

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)

    def list2hot(self, label):
        res = []
        # for i in ['Gussian','Gamma','Rayleign','Exponential','Poisson','Uniform']:
        for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def list2hot_f(self, label):
        res = []
        for i in ['Kalman', 'Wiener']:
            if i in label:
                res.append(1)
            else:
                res.append(0)
        return res

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]

    def __len__(self):
        return self.X.size(0)


def read_pickle(file_name):
    res = None
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            res = pickle.load(f)
    return res


def one_class_labeling_sz(labels, abnormal_class: int, seed):
    normal_idx = np.where(labels != abnormal_class)[0]
    abnormal_idx = np.where(labels == abnormal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


def one_class_labeling(labels, normal_class: int, seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)

    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


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

    return train_normal_data, test_normal_data, test_abnormal_data


def load_data(root, len_num=1000, seed=1024):
    all_normal_patient, all_abnormal_patient, \
        all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []

    np.random.seed(seed)
    np.random.shuffle(all_normal_patient)

    for i in all_normal_patient[:40]:  # 41
        np.random.seed(seed)
        np.random.shuffle(i)
        train_data.append(i[:600, :, :len_num])

    test_normal_data = []
    test_abnormal_data = []
    val_normal_data = []
    val_abnormal_data = []
    test_all_normal_patient = all_normal_patient[40:]
    # 随机病人
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
        # 随机单个病人时间段
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
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:100, :, :len_num])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    val_normal_data = np.concatenate(val_normal_data, axis=0)
    val_abnormal_data = np.concatenate(val_abnormal_data, axis=0)

    return train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data


def load_data_J(root, len_num=1000, seed=1024):
    all_normal_patient, all_abnormal_patient, \
        all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []

    np.random.seed(seed)
    np.random.shuffle(all_normal_patient)

    for i in all_normal_patient[:40]:  # 41
        np.random.seed(seed)
        np.random.shuffle(i)
        train_data.append(i[:600, :, :len_num])

    test_normal_data = []
    test_abnormal_data = []
    val_normal_data = []
    val_abnormal_data = []
    test_all_normal_patient = all_normal_patient[40:]
    # 随机病人
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
        # 随机单个病人时间段
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100
        train_data.append(i[100:200, :, :len_num])

    for i in all_normal_abnormal_patient_A[:11]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100
        train_data.append(i[100:200, :, :len_num])

    for i in test_all_normal_patient[:5]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:400, :, :len_num])  # 400
    # k = 0
    for i in all_normal_abnormal_patient_N[:11]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:100, :, :len_num])
    # test
    for i in all_abnormal_patient[15:]:  # 31
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100
        train_data.append(i[100:200, :, :len_num])

    for i in all_normal_abnormal_patient_A[11:]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100
        train_data.append(i[100:200, :, :len_num])

    for i in test_all_normal_patient[5:]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:400, :, :len_num])  # 400

    for i in all_normal_abnormal_patient_N[11:]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:100, :, :len_num])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    val_normal_data = np.concatenate(val_normal_data, axis=0)
    val_abnormal_data = np.concatenate(val_abnormal_data, axis=0)

    return train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data


def load_data_deepsvdd(root, len_num=1000, seed=1024):
    all_normal_patient, all_abnormal_patient, \
        all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []

    test_normal_data = []
    test_abnormal_data = []
    val_normal_data = []
    val_abnormal_data = []

    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_A)
    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_N)

    for i in all_normal_patient[:40]:  # 41
        np.random.seed(seed)
        np.random.shuffle(i)
        # train_data.append(i[:600, :, :len_num])
        train_data.append(i[:300, :, :len_num])

    for i in all_normal_abnormal_patient_N[:23]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        train_data.append(i[:50, :, :len_num])  # 100

    for i in all_normal_patient[0:40]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[300:450, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_N[0:23]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[50:100, :, :len_num])  # 100

    for i in all_normal_patient[0:40]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[450:600, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_N[0:23]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[100:150, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[:23]:  # 23
        len = i.shape[0]
        med = int(len / 2)
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:med, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[:23]:  # 23
        len = i.shape[0]
        med = int(len / 2)
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[med:, :, :len_num])  # 100

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    val_normal_data = np.concatenate(val_normal_data, axis=0)
    val_abnormal_data = np.concatenate(val_abnormal_data, axis=0)
    return train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data


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


def data_2_dataset(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, opt):
    global train_X, val_X, test_X
    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)
    # train_X = train_data[:128]
    # val_X = val_data[:128]
    # test_X = test_data[:128]

    # train_X = train_data[:1000]
    # val_X = val_data[:500]
    # test_X = test_data[:500]
    #
    # train_label = train_label[:1000]
    # val_label = val_label[:500]
    # test_label = test_label[:500]

    train_X = train_data
    val_X = val_data
    test_X = test_data

    # if opt.normalize:  # 归一化
    #     print("[INFO] Data Normalization!")
    #     # Normalize
    #     x_train_max = np.max(train_data)
    #     x_train_min = np.min(train_data)
    #     # train_X = 2. * (train_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
    #     # # Test is secret
    #     # val_X = 2. * (val_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
    #     # test_X = 2. * (test_data - x_train_min) / (x_train_max - x_train_min) - 1.  # 线性归一化 [-1,1]
    #     train_X = (train_data - x_train_min) / (x_train_max - x_train_min)  # 线性归一化 [0,1]
    #     val_X = (val_data - x_train_min) / (x_train_max - x_train_min)  # 线性归一化 [-1,1]
    #     test_X = (test_data - x_train_min) / (x_train_max - x_train_min)   # 线性归一化 [-1,1]
    #     # test
    #     # x_val_max = np.max(val_data)
    #     # x_val_min = np.min(val_data)
    #     # x_test_max = np.max(test_data)
    #     # x_test_min = np.min(test_data)
    #     # x_max=max(x_val_max,x_test_max)
    #     # x_min=max(x_val_min,x_test_min)
    #     # val_X = (val_data - x_min) / (x_max - x_min)  # 线性归一化 [-1,1]
    #     # test_X = (test_data - x_min) / (x_max - x_min) # 线性归一化 [-1,1]
    # train_X=np.concatenate([train_X,train_X[::-1]],axis=0)
    # train_X1=copy.deepcopy(train_X)
    # for i in range(train_X1.shape[0]):
    #     train_X1[i][0]=train_X1[i][0][::-1]
    #     train_X1[i][1]=train_X1[i][1][::-1]
    # train_X = np.concatenate([train_X, train_X1], axis=0)
    # train_label=np.concatenate([train_label,train_label[::-1]],axis=0)
    if opt.model in ['AE_CNN_self', 'AE_CNN_self_2', 'AE_CNN_self_3', 'AE_CNN_self_4', 'AE_CNN_self_5',
                     'AE_CNN_self_6', 'AE_CNN_self_10', 'AE_CNN_self_33', 'AE_CNN_self_66', 'AE_CNN_self_661',
                     'AE_CNN_self_41', 'AE_CNN_self_42']:
        train_dataset = TransformDataset1(train_X, train_label)
        val_dataset = TransformDataset1(val_X, val_label)
        test_dataset = TransformDataset1(test_X, test_label)
        print()
    elif opt.model in ['AE_CNN_self_7', 'AE_CNN_self_8', 'AE_CNN_self_9', 'AE_CNN_self_11', 'AE_CNN_self_12',
                       'AE_CNN_self_77', 'AE_CNN_self_88', 'AE_CNN_self_888', 'AE_CNN_self_8888', 'AE_CNN_self_99',
                       'AE_CNN_self_1010', 'AE_CNN_self_44']:
        train_dataset = TransformDataset(train_X, train_label)
        val_dataset = TransformDataset(val_X, val_label)
        test_dataset = TransformDataset(test_X, test_label)
        print()
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


def data_3_dataset(train_J, train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, opt):
    global train_X, val_X, test_X
    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)
    train_X = train_data[:train_J.shape[0]]
    val_X = val_data
    test_X = test_data

    train_dataset = RawDataset2(train_J, train_X, train_label)
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


def test_data(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, opt):
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
        test_X = 2. * (test_data - x_train_min) / (x_train_max - x_train_min) - 1.

    train_dataset = TransformDataset(train_X, train_label)


def load_ucr(opt, dataset_name):
    if dataset_name in ['MFPT']:
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

    elif dataset_name in ['SWAT']:

        normal = pd.read_csv(
            "/home/hanhan/dataset/SWAT/SWaT_Dataset_Normal_v1.csv")  # , nrows=1000)
        normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
        print(normal.shape)

        for i in list(normal):
            normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype('float32')
        data_N = normal.values
        label_N = np.zeros(len(data_N))

        # Read data
        attack = pd.read_csv("/home/hanhan/dataset/SWAT/SWaT_Dataset_Attack_v0.csv",
                             sep=";")  # , nrows=1000)
        labels = [float(label != 'Normal') for label in attack["Normal/Attack"].values]
        attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
        print(attack.shape)

        # Transform all columns into float64
        for i in list(attack):
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype('float32')
        data_A = attack.values
        label_A = np.ones(len(data_A))

        data_X = np.concatenate((data_N, data_A))
        data_Y = np.concatenate((label_N, label_A))

    elif dataset_name in ['WADI']:

        normal = pd.read_csv("/home/huangxunhua/ECG_platform/KalmanAE/experiments/data/MADI/WADI_14days.csv", sep=',',
                             skiprows=[0, 1, 2, 3], skip_blank_lines=True)  # , nrows=1000)
        normal = normal.drop(normal.columns[[0, 1, 2, 50, 51, 86, 87]],
                             axis=1)  # drop the empty columns and the date/time columns
        # Downsampling
        normal = normal.groupby(np.arange(len(normal.index)) // 5).mean()

        normal.isnull().sum().sum()
        normal = normal.fillna(0)
        print(normal.shape)

        normal = normal.astype('float32')
        data_N = normal.values
        label_N = np.zeros(len(data_N))

        # Read data
        attack = pd.read_csv("/home/huangxunhua/ECG_platform/KalmanAE/experiments/data/MADI/WADI_attackdata.csv",
                             sep=",")  # , nrows=1000)
        # labels = []
        #
        attack.reset_index()
        attack = attack.drop(attack.columns[[0, 1, 2, 50, 51, 86, 87]], axis=1)  # Drop the empty and date/time columns

        # Downsampling the attack data
        attack = attack.groupby(np.arange(len(attack.index)) // 5).mean()
        print(attack.shape)

        attack = attack.astype('float32')
        data_A = attack.values
        # data_N = data_N[:100000]
        # label_N = label_N[:100000]
        label_A = np.ones(len(data_A))

        data_X = np.concatenate((data_N, data_A))
        data_Y = np.concatenate((label_N, label_A))

    else:

        train_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TRAIN.tsv')),
                                delimiter='\t')  #
        test_data = np.loadtxt(os.path.join(opt.data_UCR, dataset_name, (dataset_name + '_TEST.tsv')),
                               delimiter='\t')  #

        data_ALL = np.concatenate((train_data, test_data), axis=0)
        data_X = data_ALL[:, 1:]  # (16637,96)
        data_Y = data_ALL[:, 0] - min(data_ALL[:, 0])  # (16637,)

    ###anomaly detection
    label_idxs = np.unique(data_Y)
    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(data_Y == idx)[0])

    if opt.normal_idx >= len(label_idxs):
        normal_idx = opt.normal_idx % len(label_idxs)
    else:
        normal_idx = opt.normal_idx
    print("[Stat] All class: {}".format(class_stat))

    if dataset_name == 'EpilepticSeizure':
        labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)

    else:
        labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)

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

    # val_N_X_len = val_N_X.shape[0]
    # test_N_X_len = test_N_X.shape[0]
    data_A_X_len = data_A_X.shape[0]

    # Split abnormal samples
    # data_A_X_idx = list(range(data_A_X_len))
    # # np.random.shuffle(data_A_X_idx)
    # np.random.shuffle(data_A_X)
    # val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    # val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    #
    # np.random.shuffle(data_A_X)
    # test_A_X = data_A_X[data_A_X_idx[:test_N_X_len]]
    # test_A_Y = data_A_Y[data_A_X_idx[:test_N_X_len]]
    ## val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    ## val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    ## test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    ## test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    ####### 正常与异常不平衡，采用异常全用的原则###########

    val_N_X_len = data_A_X_len // 2
    test_N_X_len = data_A_X_len // 2
    # val_N_X_len = data_A_X_len // 3
    # test_N_X_len = data_A_X_len // 3

    data_A_X_idx = list(range(data_A_X_len))
    val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    ####### 正常与异常平衡，采用异常不全用的原则###########

    # val_N_X_len = val_N_X.shape[0]
    # test_N_X_len = test_N_X.shape[0]
    # data_A_X_idx = list(range(data_A_X_len))
    # val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    # val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    # test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    # test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    val_X = np.concatenate((val_N_X, val_A_X))
    val_Y = np.concatenate((val_N_Y, val_A_Y))
    test_X = np.concatenate((test_N_X, test_A_X))
    test_Y = np.concatenate((test_N_Y, test_A_Y))

    train_X = np.expand_dims(train_X, 1)
    val_X = np.expand_dims(val_X, 1)
    test_X = np.expand_dims(test_X, 1)

    if opt.model in ['AE_CNN_self', 'AE_CNN_self_2', 'AE_CNN_self_3', 'AE_CNN_self_4', 'AE_CNN_self_5',
                     'AE_CNN_self_6', 'AE_CNN_self_10', 'AE_CNN_self_33', 'AE_CNN_self_666',
                     'AE_CNN_self_661']:
        train_dataset = TransformDataset1(train_X, train_Y, snr=opt.Snr)
        val_dataset = TransformDataset1(val_X, val_Y, snr=opt.Snr)
        test_dataset = TransformDataset1(test_X, test_Y, snr=opt.Snr)
        print()

    elif opt.model in ['AE_CNN_self_7', 'AE_CNN_self_8', 'AE_CNN_self_9', 'AE_CNN_self_11', 'AE_CNN_self_12',
                       'AE_CNN_self_77', 'AE_CNN_self_88', 'AE_CNN_self_99', 'AE_CNN_self_1010', 'AE_CNN_self_44',
                       'AE_CNN_self_888', 'AE_CNN_self_8888', 'AE_CNN_self_final', 'AE_CNN_self_66']:
        if opt.filter_type == 'l1':
            train_dataset = TransformDataset_l1(train_X, train_Y, opt, snr=opt.Snr, train=1)
            val_dataset = TransformDataset_l1(val_X, val_Y, opt, snr=opt.Snr, train=2)
            test_dataset = TransformDataset_l1(test_X, test_Y, opt, snr=opt.Snr, train=3)
            print()
        elif opt.filter_type == 'hp':
            train_dataset = TransformDataset_hp(train_X, train_Y, opt, snr=opt.Snr, train=1)
            val_dataset = TransformDataset_hp(val_X, val_Y, opt, snr=opt.Snr, train=2)
            test_dataset = TransformDataset_hp(test_X, test_Y, opt, snr=opt.Snr, train=3)
            print()
        else:
            train_dataset = TransformDataset2(train_X, train_Y, opt, snr=opt.Snr, train=1)
            val_dataset = TransformDataset2(val_X, val_Y, opt, snr=opt.Snr, train=2)
            test_dataset = TransformDataset2(test_X, test_Y, opt, snr=opt.Snr, train=3)
            print()
    else:
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
            dataset=test_dataset,  # torch TensorDataset format val_dataset
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True)
    }

    return dataloader, train_X.shape[-1]
