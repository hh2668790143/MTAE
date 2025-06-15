import os
import pickle

import neurokit2 as nk
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ecg_dataset.HRVTool import getP, getP_s
from ecg_dataset.getHRV import get_nni_24H_1, get_grv_list
from ecg_dataset.get_FN_data import FN_process
from ecg_dataset.load_data import load_cpsc, load_icentia11k, load_ptbxl, load_chaosuan, load_iridia_data, \
    load_IRIDIA_AF, load_cpsc_cls, load_icentia11k_cls
from ecg_dataset.data_utils import down_sample, time_warp, preprocess_signals


from model.lib_for_SLMR.utils import noise_mask

from options import Options



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


def one_class_labeling(labels, normal_class: int, seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)

    np.random.shuffle(normal_idx)
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


def mask_waves3(ecg, maskwhere):
    try:
        rpeaks = (nk.ecg_peaks(ecg, sampling_rate=100)[1]['ECG_R_Peaks']).astype(int)
    except Exception as e:
        rpeaks = []
        print(e)
    mask = ['Q', 'ST', 'T']
    try:
        # _, waves_peak = nk.ecg_delineate(ecg, rpeaks, sampling_rate=100, method="dwt", show=False, show_type='all')
        # p_wave = waves_peak['ECG_P_Peaks']
        if maskwhere == 'Q':
            for rpeak in rpeaks:
                if rpeak - 20 >= 0:
                    ecg[rpeak - 20:rpeak] = [(ecg[rpeak - 20] + ecg[rpeak]) / 2] * 20
                else:
                    ecg[0:rpeak] = [(ecg[0] + ecg[rpeak]) / 2] * rpeak
        elif maskwhere == 'ST':
            for rpeak in rpeaks:
                if rpeak + 20 < len(ecg):
                    ecg[rpeak:rpeak + 20] = [(ecg[rpeak] + ecg[rpeak + 20]) / 2] * 20
                else:
                    ecg[rpeak:len(ecg)] = np.array([(ecg[-1] + ecg[rpeak]) / 2] * (len(ecg) - rpeak))
        elif maskwhere == 'T':
            for rpeak in rpeaks:
                if rpeak + 20 < len(ecg) and rpeak + 35 < len(ecg):
                    ecg[rpeak + 20:rpeak + 35] = [(ecg[rpeak + 35] + ecg[rpeak + 20]) / 2] * 15
                elif rpeak + 20 < len(ecg) and rpeak + 35 >= len(ecg):
                    ecg[rpeak + 20:len(ecg)] = [(ecg[-1] + ecg[rpeak + 20]) / 2] * (len(ecg) - rpeak - 20)
                # else:
                #     print('无可mask的T波')
        elif maskwhere == 'random':
            for rpeak in rpeaks:
                mask_ = np.random.choice(mask)
                if mask_ == 'Q':
                    if rpeak - 20 >= 0:
                        ecg[rpeak - 20:rpeak] = [(ecg[rpeak - 20] + ecg[rpeak]) / 2] * 20
                    else:
                        ecg[0:rpeak] = [(ecg[0] + ecg[rpeak]) / 2] * rpeak
                elif mask_ == 'ST':
                    if rpeak + 20 < len(ecg):
                        ecg[rpeak:rpeak + 20] = [(ecg[rpeak] + ecg[rpeak + 20]) / 2] * 20

                    else:
                        ecg[rpeak:len(ecg)] = np.array([(ecg[-1] + ecg[rpeak]) / 2] * (len(ecg) - rpeak))
                else:
                    if rpeak + 20 < len(ecg) and rpeak + 35 < len(ecg):
                        ecg[rpeak + 20:rpeak + 35] = [(ecg[rpeak + 35] + ecg[rpeak + 20]) / 2] * (35 - 20)

                    elif rpeak + 20 < len(ecg) and rpeak + 35 >= len(ecg):
                        ecg[rpeak + 20:len(ecg)] = [(ecg[-1] + ecg[rpeak + 20]) / 2] * (len(ecg) - rpeak - 20)
                        # else:
                        #     print('无可mask的T波')
    except Exception as e:
        print(e)

    return ecg


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


class RawDataset_tl(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        # X = X[:, :2, :]
        X = X[:, :1, :]

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


class LorentzDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        np.zeros_like(x)

        x_lr = []
        x_sdnn = []
        x_rmssd = []

        x_lr_1 = []
        x_lr_2 = []
        x_lr_3 = []

        x_sdnn_1 = []
        x_rmssd_1 = []

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                lr, sdnn, rmssd = get_nni_24H_1(x[i][j], 100)
                lr_1 = lr[0]
                lr_2 = lr[1]
                x_lr_1 = x_lr_1 + lr_1
                x_lr_2 = x_lr_2 + lr_2

                # draw_LR(x_lr_1, x_lr_2)

                x_sdnn_1.append(sdnn)
                x_rmssd_1.append(rmssd)

            x_lr_1 += [0 for i in range(np.size(x, 2) - len(x_lr_1))]
            x_lr_2 += [0 for i in range(np.size(x, 2) - len(x_lr_2))]

            x_lr_3.append(x_lr_1)
            x_lr_3.append(x_lr_2)

            x_lr.append(x_lr_3)
            x_sdnn.append(x_sdnn_1)
            x_rmssd.append(x_rmssd_1)

            x_lr_1 = []
            x_lr_2 = []
            x_lr_3 = []
            x_sdnn_1 = []
            x_rmssd_1 = []

        x_lr = np.asarray(x_lr, dtype=np.float32)
        x_sdnn = np.asarray(x_sdnn, dtype=np.float32)
        x_rmssd = np.asarray(x_rmssd, dtype=np.float32)
        x_sdnn = np.mean(x_sdnn, 1)
        x_rmssd = np.mean(x_rmssd, 1)
        x_sd = np.asarray(list(zip(x_sdnn, x_rmssd)), dtype=np.float32)
        x_sd = np.expand_dims(x_sd, 2).repeat(1000, axis=2)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_lr = torch.Tensor(x_lr)
        self.x_sdnn = torch.Tensor(x_sdnn)
        self.x_rmssd = torch.Tensor(x_rmssd)
        self.x_sd = torch.Tensor(x_sd)

    def __getitem__(self, index):

        return self.x[index], self.x_lr[index], self.x_sd[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class HrvDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        np.zeros_like(x)
        if len(x[1]) == 1:
            hrv = [[]]
        elif len(x[1]) == 2:
            hrv = [[], []]

        x_hrv = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                hrv_list = get_grv_list(x[i][j], 100)
                hrv[j] = hrv_list
            x_hrv.append(hrv)
        x_hrv = np.asarray(x_hrv, dtype=np.float32)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_hrv = torch.Tensor(x_hrv)

    def __getitem__(self, index):

        return self.x[index], self.x_hrv[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FftDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_fft = torch.Tensor(x_fft)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class MaskDataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('cpsc2021_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('icentia11k_{}.npy'.format(opt.seed))
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('ptbxl_{}.npy'.format(opt.seed))
        else:
            x_mask = getP(x, opt.dataset, opt.seed)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class ChaosuankDataset(data.Dataset):
    def __init__(self, x, y, x_a, opt):
        x = np.asarray(x, dtype=np.float32)
        x_a = np.asarray(x_a, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_a)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('cpsc2021_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('icentia11k_{}.npy'.format(opt.seed))
                x_mask = np.expand_dims(x_mask, axis=1)
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('ptbxl_{}.npy'.format(opt.seed))
        else:
            x_mask = getP(x, opt.dataset, opt.seed)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('cpsc2021_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('icentia11k_{}.npy'.format(opt.seed))
                x_mask = np.expand_dims(x_mask, axis=1)
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('ptbxl_{}.npy'.format(opt.seed))
        else:
            x_mask = getP(x, opt.dataset, opt.seed)

        if opt.isDataProcessed:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('fmds_data_npy/cpsc2021_dp_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('fmds_data_npy/icentia11k_dp_{}.npy'.format(opt.seed))
                # x_mask = np.expand_dims(x_mask, axis=1)
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('fmds_data_npy/ptbxl_dp_{}.npy'.format(opt.seed))
        else:
            oncatenated = np.concatenate((x_mask, x_mask), axis=2)
            for i in range(oncatenated.shape[0]):
                for j in range(oncatenated.shape[1]):
                    x_mask[i][j] = down_sample(oncatenated[i][j], 2000, 1700)[:1000]

            for i in range(x_mask.shape[0]):
                input = np.expand_dims(x_mask[i], 0)
                input = input.transpose(0, 2, 1)
                # input = input.permute(0, 2, 1)
                input_w = time_warp(input)
                input_w = input_w.transpose(0, 2, 1)
                input_w = np.squeeze(input_w)
                x_mask[i] = input_w

            np.save('fmds_data_npy/{}_dp_'.format(opt.dataset) + str(opt.seed), x_mask)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        # coeffs = pywt.wavedec(x, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        # x_fft = coeffs[0]
        #
        # coeffs_m = pywt.wavedec(x_mask, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        # x_fft_m = coeffs_m[0]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_x(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('mask', opt.dataset, opt.section, opt.seed))
        else:
            x_mask = getP_s(x, opt.dataset, opt.seed, opt.section)

        if opt.isDataProcessed:
            x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('dataProcess', opt.dataset, opt.section, opt.seed))
        else:
            x_mask_t = np.concatenate((x_mask, x_mask), axis=2)
            for i in range(x_mask_t.shape[0]):
                for j in range(x_mask_t.shape[1]):
                    x_mask[i][j] = down_sample(x_mask_t[i][j], 100, 85)[:1000]

            for i in range(x_mask.shape[0]):
                input = np.expand_dims(x_mask[i], 0)
                input = input.transpose(0, 2, 1)
                input_w = time_warp(input)
                input_w = input_w.transpose(0, 2, 1)
                input_w = np.squeeze(input_w)
                x_mask[i] = input_w

            if not os.path.exists(
                    "./AFPD/{}/{}/{}".format('dataProcess', opt.dataset, opt.section)):
                os.makedirs("./AFPD/{}/{}/{}".format('dataProcess', opt.dataset, opt.section))

            np.save("./AFPD/{}/{}/{}/{}.npy".format('dataProcess', opt.dataset, opt.section, opt.seed), x_mask)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_tl(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('mask', opt.test_dataset, opt.section, opt.seed))
        else:
            x_mask = getP_s(x, opt.dataset, opt.seed, opt.section)

        if opt.isDataProcessed:
            x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('dataProcess', opt.test_dataset, opt.section, opt.seed))
        else:
            x_mask_t = np.concatenate((x_mask, x_mask), axis=2)
            for i in range(x_mask_t.shape[0]):
                for j in range(x_mask_t.shape[1]):
                    x_mask[i][j] = down_sample(x_mask_t[i][j], 2000, 1700)[:1000]

            for i in range(x_mask.shape[0]):
                input = np.expand_dims(x_mask[i], 0)
                input = input.transpose(0, 2, 1)
                input_w = time_warp(input)
                input_w = input_w.transpose(0, 2, 1)
                input_w = np.squeeze(input_w)
                x_mask[i] = input_w

            if not os.path.exists(
                    "./AFPD/{}/{}/{}".format('dataProcess', opt.dataset, opt.section)):
                os.makedirs("./AFPD/{}/{}/{}".format('dataProcess', opt.dataset, opt.section))

            np.save("./AFPD/{}/{}/{}/{}.npy".format('dataProcess', opt.dataset, opt.section, opt.seed), x_mask)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        x = x[:, :1, :]
        x_mask = x_mask[:, :1, :]
        x_fft = x_fft[:, :1, :]
        x_fft_m = x_fft_m[:, :1, :]

        # x = x[:, :2, :]
        # x_mask = x_mask[:, :2, :]
        # x_fft = x_fft[:, :2, :]
        # x_fft_m = x_fft_m[:, :2, :]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_xx(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('dataProcessP', opt.dataset, opt.section, opt.seed))

        if opt.is_standardization:
            x = preprocess_signals(x)
            x_mask = preprocess_signals(x_mask)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_cls(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('dataProcessP', opt.dataset, opt.section, opt.seed))
        y_mask = np.ones(x_mask.shape[0])

        x = np.concatenate((x, x_mask), axis=0)
        y = np.concatenate((y, y_mask), axis=0)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_tlx(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_mask = np.load("./AFPD/{}/{}/{}/{}.npy".format('dataProcessP', opt.test_dataset, opt.section, opt.seed))

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        if opt.dataset == 'cpsc2021':
            x = x[:, :2, :]
            x_mask = x_mask[:, :2, :]
            x_fft = x_fft[:, :2, :]
            x_fft_m = x_fft_m[:, :2, :]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset_tlxx(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        if opt.dataset == 'cpsc2021':
            x = x[:, :2, :]
            x_fft = x_fft[:, :2, :]
        else:
            x = x[:, :1, :]
            x_fft = x_fft[:, :1, :]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSNODataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if opt.is_standardization:
            x = preprocess_signals(x)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FNDataset(data.Dataset):
     def __init__(self, x, y, opt, section):
        opt.section = section

        if opt.isDataProcessed:

            with open("/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}/{}_{}.pkl".format('FN', opt.dataset, opt.section, str(opt.snr), str(opt.seed)), 'rb') as f:
                data = pickle.load(f)

        else:

            fix_data, fix_noisy_data, labels, fix_data_f, fix_noisy_data_f, labels_f = FN_process(x, opt.snr)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f

            if not os.path.exists(
                    "/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}".format('FN', opt.dataset, opt.section)):
                os.makedirs("/home/hanhan/workspace/Han/My/MTAE/processed_data/{}/{}/{}".format('FN', opt.dataset, opt.section))

            with open("/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}/{}_{}.pkl".format('FN', opt.dataset, opt.section, str(opt.snr), str(opt.seed)), 'wb') as f:
                pickle.dump(data, f)

        fix_data = data['fix_data']
        fix_noisy_data = data['fix_noisy_data']
        labels = data['labels']
        fix_data_f = data['fix_data_f']
        fix_noisy_data_f = data['fix_noisy_data_f']
        labels_f = data['labels_f']

        fix_data = np.array(fix_data)
        fix_noisy_data = np.array(fix_noisy_data)
        labels = np.array(labels)
        fix_data_f = np.array(fix_data_f)
        fix_noisy_data_f = np.array(fix_noisy_data_f)
        labels_f = np.array(labels_f)
        
        self.X = torch.Tensor(x)
        self.Y = torch.Tensor(y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)
     def __getitem__(self, index):
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]
     def __len__(self):
        return self.X.size(0)


class FNDataset_tl(data.Dataset):
    def __init__(self, x, y, opt, section):
        opt.section = section

        if opt.isDataProcessed:

            with open("/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}/{}_{}.pkl".format('FN', opt.test_dataset,
                                                                                                opt.section,
                                                                                                str(opt.snr),
                                                                                                str(opt.seed)),
                      'rb') as f:
                data = pickle.load(f)

        else:

            fix_data, fix_noisy_data, labels, fix_data_f, fix_noisy_data_f, labels_f = FN_process(x, opt.snr)

            data = {}
            data['fix_data'] = fix_data
            data['fix_noisy_data'] = fix_noisy_data
            data['labels'] = labels
            data['fix_data_f'] = fix_data_f
            data['fix_noisy_data_f'] = fix_noisy_data_f
            data['labels_f'] = labels_f

            if not os.path.exists(
                    "/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}".format('FN', opt.test_dataset, opt.section)):
                os.makedirs(
                    "/home/hanhan/workspace/Han/My/MTAE/processed_data/{}/{}/{}".format('FN', opt.test_dataset, opt.section))

            with open("/home/hanhan/workspace/My/MTAE/processed_data/{}/{}/{}/{}_{}.pkl".format('FN', opt.test_dataset,
                                                                                                opt.section,
                                                                                                str(opt.snr),
                                                                                                str(opt.seed)),
                      'wb') as f:
                pickle.dump(data, f)

        fix_data = data['fix_data']
        fix_noisy_data = data['fix_noisy_data']
        labels = data['labels']
        fix_data_f = data['fix_data_f']
        fix_noisy_data_f = data['fix_noisy_data_f']
        labels_f = data['labels_f']

        fix_data = np.array(fix_data)
        fix_noisy_data = np.array(fix_noisy_data)
        labels = np.array(labels)
        fix_data_f = np.array(fix_data_f)
        fix_noisy_data_f = np.array(fix_noisy_data_f)
        labels_f = np.array(labels_f)

        x = x[:, :1, :]
        fix_data = fix_data[:, :1, :]
        fix_noisy_data = fix_noisy_data[:, :1, :]
        fix_data_f = fix_data_f[:, :1, :]
        fix_noisy_data_f = fix_noisy_data_f[:, :1, :]

        self.X = torch.Tensor(x)
        self.Y = torch.Tensor(y)
        self.fix_data = torch.Tensor(fix_data)
        self.fix_noisy_data = torch.Tensor(fix_noisy_data)
        self.labels = torch.Tensor(labels)

        self.fix_data_f = torch.Tensor(fix_data_f)
        self.fix_noisy_data_f = torch.Tensor(fix_noisy_data_f)
        self.labels_f = torch.Tensor(labels_f)

    def __getitem__(self, index):
        return self.X[index], self.fix_data[index], self.Y[index], self.fix_noisy_data[index], self.labels[index], \
            self.fix_data_f[index], self.fix_noisy_data_f[index], self.labels_f[index]

    def __len__(self):
        return self.X.size(0)

class SlidingWindowDataset(data.Dataset):
    def __init__(self, opt, data, label, window, target_dim=None, horizon=1):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.window = window  # window = X_length
        self.target_dim = target_dim  # target_dim = nc
        self.horizon = horizon
        self.opt = opt
        self.cut_len = int(self.data.shape[1] * 0.9)
        if self.cut_len % 2 != 0: self.cut_len += 1

    def __getitem__(self, index):
        x_data = self.data[index]

        x = x_data[:self.cut_len, ]
        # x_mask = zero_one(x)
        # x_mask = x * x_mask
        mask = noise_mask(x, masking_ratio=self.opt.mask_ratio, lm=3, mode='separate', distribution='geometric',
                          exclude_feats=None)
        mask = torch.from_numpy(mask) * x
        y = x_data[self.cut_len:, ]
        return x, mask, y, self.label[index]

    def __len__(self):
        # return len(self.data) - self.window
        return len(self.data)



def get_dataloader(opt):
    global train_dataset, val_dataset, test_dataset

    if opt.dataset == 'cpsc2021':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc(opt)
    elif opt.dataset == 'icentia11k':
        train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k(opt)
    elif opt.dataset == 'ptbxl':
        train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)
    elif opt.dataset == 'chaosuan':
        train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal = load_chaosuan(opt)
    elif opt.dataset == 'IRIDIA_AF':
        train_data, train_label, val_data, val_label, test_data, test_label = load_IRIDIA_AF(opt)

    if opt.is_all_data:
        train_X = train_data
        train_Y = train_label
        val_X = val_data
        val_Y = val_label
        test_X = test_data
        test_Y = test_label
    else:
        train_X = train_data[:500, :, :]
        train_Y = train_label[:500]
        val_X = val_data[:200, :, :]
        val_Y = val_label[:200]
        test_X = test_data[:200, :, :]
        test_Y = test_label[:200]

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.augmentation == "mask":
        train_dataset = MaskDataset(train_X, train_Y, opt)
        val_dataset = MaskDataset(val_X, val_Y, opt)
        test_dataset = MaskDataset(test_X, test_Y, opt)
    elif opt.augmentation == "hrv":
        train_dataset = HrvDataset(train_X, train_Y)
        val_dataset = HrvDataset(val_X, val_Y)
        test_dataset = HrvDataset(test_X, test_Y)
    elif opt.augmentation == "lr":
        train_dataset = LorentzDataset(train_X, train_Y)
        val_dataset = LorentzDataset(val_X, val_Y)
        test_dataset = LorentzDataset(test_X, test_Y)
    elif opt.augmentation == "fft":
        train_dataset = FftDataset(train_X, train_Y)
        val_dataset = FftDataset(val_X, val_Y)
        test_dataset = FftDataset(test_X, test_Y)
    elif opt.augmentation == "fft_mask":
        train_dataset = FMDataset(train_X, train_Y, opt)
        val_dataset = FMDataset(val_X, val_Y, opt)
        test_dataset = FMDataset(test_X, test_Y, opt)
    elif opt.augmentation == "fft_mask_downSample":
        train_dataset = FMDSDataset(train_X, train_Y, opt)
        val_dataset = FMDSDataset(val_X, val_Y, opt)
        test_dataset = FMDSDataset(test_X, test_Y, opt)
    elif opt.augmentation == "chaosuan":
        train_dataset = ChaosuankDataset(train_X, train_Y, train_data_abnormal, opt)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    elif opt.augmentation == "fft_mask_downSample_x":
        train_dataset = FMDSDataset_x(train_X, train_Y, opt, "train")
        # val_dataset = FMDSDataset_x(val_X, val_Y, opt, "val")
        # test_dataset = FMDSDataset_x(test_X, test_Y, opt, "test")
        val_dataset = FMDSNODataset(val_X, val_Y, opt)
        test_dataset = FMDSNODataset(test_X, test_Y, opt)
    elif opt.augmentation == "fft_mask_downSample_xx":
        train_dataset = FMDSDataset_xx(train_X, train_Y, opt, "train")
        val_dataset = FMDSNODataset(val_X, val_Y, opt)
        test_dataset = FMDSNODataset(test_X, test_Y, opt)
    elif opt.augmentation == "fft_mask_downSample_tl":
        train_dataset = FMDSDataset_x(train_X, train_Y, opt, "train")
        val_dataset = FMDSDataset_x(val_X, val_Y, opt, "val")
        test_dataset = FMDSDataset_x(test_X, test_Y, opt, "test")
    elif opt.augmentation == "Filter_Noise":
        train_dataset = FNDataset(train_X, train_Y, opt, "train")
        val_dataset = FNDataset(val_X, val_Y, opt, "val")
        test_dataset = FNDataset(test_X, test_Y, opt, "test")
    elif opt.augmentation == "SlidingWindow":
        target_dims = opt.nc
        window_size = X_length
        X_length = int(window_size * 0.9)
        if X_length % 2 != 0: X_length += 1
        opt.pre_len = window_size - X_length
        train_X = train_X.transpose(0, 2, 1)
        test_X = test_X.transpose(0, 2, 1)
        val_X = val_X.transpose(0, 2, 1)

        train_dataset = SlidingWindowDataset(opt, train_X, train_Y, window_size, target_dims)
        test_dataset = SlidingWindowDataset(opt, test_X, test_Y, window_size, target_dims, val_Y)
        val_dataset = SlidingWindowDataset(opt, val_X, val_Y, window_size, target_dims, test_Y)
    else:
        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length


def get_dataloader_tl(opt):
    global train_dataset, val_dataset, test_dataset

    if opt.test_dataset == 'cpsc2021':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc(opt)
    elif opt.test_dataset == 'icentia11k':
        train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k(opt)
    elif opt.test_dataset == 'ptbxl':
        train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)
    elif opt.test_dataset == 'chaosuan':
        train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal = load_chaosuan(opt)

    elif opt.test_dataset == 'IRIDIA_AF':
        train_data, train_label, val_data, val_label, test_data, test_label = load_IRIDIA_AF(opt)


    if opt.is_all_data:
        train_X = train_data
        train_Y = train_label
        val_X = val_data
        val_Y = val_label
        test_X = test_data
        test_Y = test_label
    else:
        train_X = train_data[:10, :, :]
        train_Y = train_label[:10]
        val_X = val_data[:200, :, :]
        val_Y = val_label[:200]
        test_X = test_data[:200, :, :]
        test_Y = test_label[:200]

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.augmentation == "Filter_Noise":
        train_dataset = FNDataset_tl(train_X, train_Y, opt, "train")
        val_dataset = FNDataset_tl(val_X, val_Y, opt, "val")
        test_dataset = FNDataset_tl(test_X, test_Y, opt, "test")
    else:
        train_dataset = RawDataset_tl(train_X, train_Y)
        val_dataset = RawDataset_tl(val_X, val_Y)
        test_dataset = RawDataset_tl(test_X, test_Y)


    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length


def get_dataloader_cls(opt):
    global train_dataset, val_dataset, test_dataset

    if opt.add_abnormal:
        if opt.dataset == 'cpsc2021':
            train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc_cls(opt)
        elif opt.dataset == 'icentia11k':
            train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k_cls(opt)
        elif opt.dataset == 'ptbxl':
            train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)
        elif opt.dataset == 'chaosuan':
            train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal = load_chaosuan(
                opt)
        elif opt.dataset == 'IRIDIA_AF':
            train_data, train_label, val_data, val_label, test_data, test_label = load_IRIDIA_AF(opt)
    else:
        if opt.dataset == 'cpsc2021':
            train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc(opt)
        elif opt.dataset == 'icentia11k':
            train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k(opt)
        elif opt.dataset == 'ptbxl':
            train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)
        elif opt.dataset == 'chaosuan':
            train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal = load_chaosuan(
                opt)
        elif opt.dataset == 'IRIDIA_AF':
            train_data, train_label, val_data, val_label, test_data, test_label = load_IRIDIA_AF(opt)

    if opt.is_all_data:
        train_X = train_data
        train_Y = train_label
        val_X = val_data
        val_Y = val_label
        test_X = test_data
        test_Y = test_label
    else:
        train_X = train_data[:10, :, :]
        train_Y = train_label[:10]
        val_X = val_data[:200, :, :]
        val_Y = val_label[:200]
        test_X = test_data[:200, :, :]
        test_Y = test_label[:200]

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.augmentation == "mask":
        train_dataset = MaskDataset(train_X, train_Y, opt)
        val_dataset = MaskDataset(val_X, val_Y, opt)
        test_dataset = MaskDataset(test_X, test_Y, opt)

    elif opt.augmentation == "fft_mask_downSample_cls":
        train_dataset = FMDSDataset_cls(train_X, train_Y, opt, "train")
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    else:
        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True),

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length


if __name__ == '__main__':
    device = torch.device("cuda:3" if
                          torch.cuda.is_available() else "cpu")
    opt = Options().parse()

    DATASETS_NAME = {
        # 'cpsc2021': 1,
        'ptbxl': 1,
        # 'icentia11k': 1,
    }

    SEEDS = [
        0, 1, 2
    ]

    for dataset_name in list(DATASETS_NAME.keys()):

        for seed in SEEDS:

            opt.seed = seed
            opt.dataset = dataset_name

            if opt.dataset == 'cpsc2021':
                train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc(opt)
            elif opt.dataset == 'icentia11k':
                train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k(opt)
            elif opt.dataset == 'ptbxl':
                train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)

            train_data_mask = getP_s(train_data, opt.dataset, opt.seed, 'train')
            val_data_mask = getP_s(val_data, opt.dataset, opt.seed, 'val')
            test_data_mask = getP_s(test_data, opt.dataset, opt.seed, 'test')
