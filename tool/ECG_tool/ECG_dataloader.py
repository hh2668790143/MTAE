import numpy as np
import pandas as pd
import torch
import matplotlib
from torch.utils.data import DataLoader
import torch.utils.data as data
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.interpolate as spi

# from dataset.normalization import normalization
# from datasets.normalization_ecg import apply_standardizer_ecg

matplotlib.use('Agg')
# LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2"]
import matplotlib.pyplot as plt


def down_sample(signal, sample_rate_1, sample_rate_2):  # sample_rate_1 ----> sample_rate_2
    '''
    :param signal:
    :param sample_rate_1:  初始采样率
    :param sample_rate_2:  需要的采样率
    :return:
    '''
    data_ds = []
    for i in range(signal.shape[0]):
        X = np.arange(0, len(signal[i]) * (1 / sample_rate_1), (1 / sample_rate_1))  # 4s  步长0.004对应250HZ  1000个点
        new_X = np.arange(0, len(signal[i]) * (1 / sample_rate_1), (1 / sample_rate_2))  # 4S  步长0.002对应500HZ  2000个点

        ipo3 = spi.splrep(X, signal[i], k=3)
        iy3 = spi.splev(new_X, ipo3)

        data_ds.append(iy3)

    data_ds = np.array(data_ds, dtype=np.float16)

    return data_ds


def plot_sample_100Hz(signal, datename=None):
    try:
        # Plot
        signal = signal[:, :]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 2000, 100)
        x_labels = np.arange(0, 20)

        # idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        idx = [1, 3, 5, 7, 2, 4, 6, 8]
        for i in range(len(signal)):
            plt.subplot(4, 2, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        plt.savefig("./Train_Plot_100Hz/{}".format(datename) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        print("Plot Success")
        return True


    except Exception as e:
        print(e)
        return False


def plot_sample_500Hz(signal, datename=None):
    try:
        # Plot
        signal = signal[:, :]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 5000, 500)
        x_labels = np.arange(0, 10)

        # idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        idx = [1, 3, 5, 7, 2, 4, 6, 8]
        for i in range(len(signal)):
            plt.subplot(4, 2, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        plt.savefig("./Train_Plot_500Hz/{}".format(datename) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        print("Plot Success")
        return True


    except Exception as e:
        print(e)
        return False


def plot_sample_D(signal, datename=None):
    try:
        # Plot
        signal = signal[:, :2000]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1990, 250)
        x_labels = np.arange(0, 8)

        idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        for i in range(len(signal)):
            plt.subplot(6, 2, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)

        fig.tight_layout()
        plt.savefig("{}".format(datename) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1


global count


def normalized(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''

    if np.max(seq) - np.min(seq) != 0:
        return (seq - np.min(seq)) / (np.max(seq) - np.min(seq))
    else:

        return (seq - np.min(seq)) / (np.max(seq) + 1)


class RawDataset(data.Dataset):
    def __init__(self, X):
        """
        """

        self.X = torch.Tensor(X)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index]

    def __len__(self):
        return self.X.size(0)


def preprocess_signals(X_train, X_validation, X_test, outputfolder=None):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    # with open(outputfolder + 'standard_scaler.pkl', 'wb') as ss_file:
    #   pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def ecg_standard_scaler(signal):
    signal_scaler = []

    for i in range(signal.shape[0]):
        scaler = MinMaxScaler()
        # a= np.expand_dims(signal[i],1)
        scaler.fit(np.expand_dims(signal[i], 1))
        signal_scaler_single = scaler.transform(np.expand_dims(signal[i], 1))
        signal_scaler.append(signal_scaler_single)

    signal_scaler = np.array(signal_scaler)
    signal_scaler = np.squeeze(signal_scaler)

    return signal_scaler



def load_test_abnormal():
    LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    data_X_N = []
    data_Y_N = []
    data_X_A = []
    data_Y_A = []

    for i in range(1, 11):
        # print(i)
        name = 'FangChan'  # ALL2  40
        # name = 'FeiDa'  # ALL2  35
        # name = 'GengSi'  #ALL3 27
        # name = 'ZuZhi'  # ALL4 100
        # name = 'ZhengChang'
        # name = 'GuoSu'  # 10
        # name = 'ShiChan'

        print('Read {}-th to {}-th {}:  {}'.format(1, 100, name, i))
        # name = 'FeiDa'

        with open('{}/YXDECG_{}_data_{}.pickle'.format('/home/huangxunhua/Data_100Hz', i * 1000, name),
                  'rb') as handle1:

            data_X_N_S = pickle.load(handle1)  # (1000,12,10000)
            data_Y_N_S = np.zeros(len(data_X_N_S))

        data_X_N_S = np.array(data_X_N_S)
        data_Y_N_S = np.array(data_Y_N_S)

        if i == 1:
            data_X_N = data_X_N_S
            data_Y_N = data_Y_N_S
        else:
            data_X_N = np.concatenate((data_X_N, data_X_N_S))
            data_Y_N = np.concatenate((data_Y_N, data_Y_N_S))

    test_abnormal_data = data_X_N[:, :2, :1000]
    # test_abnormal_data = data_X_N[:, :2, 1000:]

    test_abnormal_data = np.concatenate([test_abnormal_data], axis=0)

    return test_abnormal_data


def load_normal():
    LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    data_X_N = []
    data_Y_N = []
    data_X_A = []
    data_Y_A = []

    for i in range(1, 11):
        # print(i)
        # name = 'FangChan'  # ALL2  40
        # name = 'FeiDa'  # ALL2  35
        # name = 'GengSi'  #ALL3 27
        # name = 'ZuZhi'  # ALL4 100
        name = 'ZhengChang'
        # name = 'GuoSu'  # 10
        # name = 'ShiChan'

        print('Read {}-th to {}-th {}:  {}'.format(1, 100, name, i))
        # name = 'FeiDa'

        with open('{}/YXDECG_{}_data_{}.pickle'.format('/home/huangxunhua/Data_100Hz', i * 1000, name),
                  'rb') as handle1:

            data_X_N_S = pickle.load(handle1)  # (1000,12,10000)
            data_Y_N_S = np.zeros(len(data_X_N_S))

        data_X_N_S = np.array(data_X_N_S)
        data_Y_N_S = np.array(data_Y_N_S)

        if i == 1:
            data_X_N = data_X_N_S
            data_Y_N = data_Y_N_S
        else:
            data_X_N = np.concatenate((data_X_N, data_X_N_S))
            data_Y_N = np.concatenate((data_Y_N, data_Y_N_S))

    # normal_data = data_X_N[:, :2, :1000]
    normal_data = data_X_N[:, :2, :]

    normal_data = np.concatenate([normal_data], axis=0)

    train_normal_data = normal_data[:, :, :1000]
    test_normal_data = normal_data[:, :, 1000:]

    return train_normal_data, test_normal_data


def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index



def load_data_s(seed):
    LOAD = True
    if LOAD:

        train_normal_data, test_normal_data = load_normal()
        test_abnormal_data = load_test_abnormal()

        len_train = train_normal_data.shape[0]
        len_test_normal = test_normal_data.shape[0]
        len_test_abnormal = test_abnormal_data.shape[0]
        med_normal = int(len_test_normal / 2)
        med_abnormal = int(len_test_abnormal / 2)

        train_data = train_normal_data
        train_label = np.zeros(len_train)
        val_data = np.concatenate([test_normal_data[:med_normal, ], test_abnormal_data[:med_abnormal, ]], axis=0)
        val_label = np.concatenate([np.zeros(med_normal), np.ones(med_abnormal)], axis=0)
        test_data = np.concatenate([test_normal_data[med_normal:, ], test_abnormal_data[med_abnormal:, ]], axis=0)
        test_label = np.concatenate([np.zeros(len_test_normal - med_normal), np.ones(len_test_abnormal - med_abnormal)],
                                    axis=0)

        train_label, train_idx = shuffle_label(train_label, seed)
        train_data = train_data[train_idx]

        val_label, val_idx = shuffle_label(val_label, seed)
        val_data = val_data[val_idx]
        val_label = val_label[val_idx]

        test_label, test_idx = shuffle_label(test_label, seed)
        test_data = test_data[test_idx]
        test_label = test_label[test_idx]

        train_domain_label = [0] * len(train_label)
        val_domain_label = [0] * len(val_label)
        test_domain_label = [0] * len(test_label)

        obj = (train_data, train_domain_label, train_label, val_data, val_domain_label, val_label, test_data,
               test_domain_label, test_label)
        with open('./dataset/ecg_source.pickle', 'wb') as fw:
            pickle.dump(obj, fw)
    else:
        with open('./dataset/ecg_source.pickle', 'rb') as fw:
            (train_data, train_domain_label, train_label, val_data, val_domain_label, val_label, test_data,
             test_domain_label, test_label) = pickle.load(fw)

    return train_data, train_domain_label, train_label, val_data, val_domain_label, val_label, test_data, test_domain_label, test_label


