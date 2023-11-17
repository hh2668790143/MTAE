import numpy as np

from dataset.load_data_cpsc2021_1 import load_data_cpsc2021, load_CPSC2021_data
from dataset.load_data_icential11k import load_icentia11k_data

from dataset.load_data_ptbxl import load_ptbxl_data


def load_cpsc(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)

    # train_data = np.mean(train_data, 1)
    # val_normal_data = np.mean(val_normal_data, 1)
    # val_abnormal_data = np.mean(val_abnormal_data, 1)
    # test_normal_data = np.mean(test_normal_data, 1)
    # test_abnormal_data = np.mean(test_abnormal_data, 1)

    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label


def load_icentia11k(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
        '/data/icentia11k/', opt.seed)

    # if NOT_ALL
    if opt.model in ["ECOD"]:
        train_data = train_data
        val_normal_data = val_normal_data
        val_abnormal_data = val_abnormal_data
        test_normal_data = test_normal_data
        test_abnormal_data = test_abnormal_data
    else:
        train_data = np.expand_dims(train_data, 1)
        val_normal_data = np.expand_dims(val_normal_data, 1)
        val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
        test_normal_data = np.expand_dims(test_normal_data, 1)
        test_abnormal_data = np.expand_dims(test_abnormal_data, 1)

    # train_data = np.stack((train_data,) * 2, axis=1)
    # val_normal_data = np.stack((val_normal_data,) * 2, axis=1)
    # val_abnormal_data = np.stack((val_abnormal_data,) * 2, axis=1)
    # test_normal_data = np.stack((test_normal_data,) * 2, axis=1)
    # test_abnormal_data = np.stack((test_abnormal_data,) * 2, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label


def load_ptbxl(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_ptbxl_data(opt)

    train_data, train_label, val_data, val_label, test_data, test_label = load_CPSC2021_data(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label
