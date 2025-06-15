import glob
import os

import numpy as np

from ecg_dataset.load_data_IRIDIA_AF import load_iridia_unAF, load_iridia_AF
from ecg_dataset.load_data_chaosuan import load_chaosuan_data
from ecg_dataset.load_data_cpsc2021 import load_data_cpsc2021, load_data_cpsc2021_cls
from ecg_dataset.load_data_icential11k import load_icentia11k_data, load_icentia11k_data_cls
from ecg_dataset.load_data_ptbxl import load_ptbxl_data


def load_cpsc(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        '/home/hanhan/dataset/CPSC2021_fangchan/ALL_100HZ/', 1000, opt.seed)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_icentia11k(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
        '/repository/dataset/data_hh/icentia11k/', opt.seed)

    # if NOT_ALL
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

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)


    return train_data, train_label, val_data, val_label, test_data, test_label



def load_ptbxl(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_ptbxl_data(opt)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label


def load_IRIDIA_AF(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_iridia_data(
        '/home/luojiawei/dataset/IRIDIA_AF', opt.seed)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label



def load_chaosuan(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, train_data_abnormal = load_chaosuan_data(opt)

    # if NOT_ALL
    train_data = np.expand_dims(train_data, 1)
    val_normal_data = np.expand_dims(val_normal_data, 1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
    test_normal_data = np.expand_dims(test_normal_data, 1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, 1)
    train_data_abnormal = np.expand_dims(train_data_abnormal, 1)

    # train_data = np.stack((train_data,) * 2, axis=1)
    # val_normal_data = np.stack((val_normal_data,) * 2, axis=1)
    # val_abnormal_data = np.stack((val_abnormal_data,) * 2, axis=1)
    # test_normal_data = np.stack((test_normal_data,) * 2, axis=1)
    # test_abnormal_data = np.stack((test_abnormal_data,) * 2, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal




def load_iridia_data(data_dir,seed):
    import multiprocessing
    pool = multiprocessing.Pool(4)
    # 获取文件夹下所有病人文件的路径
    file_paths = glob.glob(os.path.join(data_dir, '*'))
    np.random.seed(seed)
    np.random.shuffle(file_paths)
    train_files = file_paths[:len(file_paths)*2//3]
    val_files = file_paths[len(file_paths)*2//3:len(file_paths)*5//6]
    test_files = file_paths[len(file_paths)*5//6:]

    train_X = load_iridia_unAF(train_files,seed,pool)

    val_normal_X = load_iridia_unAF(val_files,seed,pool)
    val_abnormal_X = load_iridia_AF(val_files,seed,pool)

    test_normal_X = load_iridia_unAF(test_files,seed,pool)
    test_abnormal_X = load_iridia_AF(test_files,seed,pool)

    print('train:{},val_A:{},val_N:{},test_A:{},test_N:{}'.format(train_X.shape[0],val_abnormal_X.shape[0],val_normal_X.shape[0],test_abnormal_X.shape[0],test_normal_X.shape[0]))


    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X



def load_cpsc_cls(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, train_abnormal_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021_cls(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000, opt.seed)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label_cls(train_data,
                                                                                             train_abnormal_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_icentia11k_cls(opt):
    train_data, train_abnormal_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data_cls(
        '/data/icentia11k/', opt.seed)

    # if NOT_ALL
    train_data = np.expand_dims(train_data, 1)
    train_abnormal_data = np.expand_dims(train_abnormal_data, 1)
    val_normal_data = np.expand_dims(val_normal_data, 1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
    test_normal_data = np.expand_dims(test_normal_data, 1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, 1)

    # train_data = np.stack((train_data,) * 2, axis=1)
    # val_normal_data = np.stack((val_normal_data,) * 2, axis=1)
    # val_abnormal_data = np.stack((val_abnormal_data,) * 2, axis=1)
    # test_normal_data = np.stack((test_normal_data,) * 2, axis=1)
    # test_abnormal_data = np.stack((test_abnormal_data,) * 2, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label_cls(train_data,
                                                                                             train_abnormal_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label



if __name__ == '__main__':
    from options import Options
    opt = Options().parse()
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)
















def get_data_label(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED):
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


def get_data_label_cls(train_normal_data, train_abnormal_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED):
    # train_normal_data, test_normal_data, test_abnormal_data = load_data(root)

    len_train_normal = train_normal_data.shape[0]
    len_train_abnormal = train_abnormal_data.shape[0]
    len_val_normal = val_normal_data.shape[0]
    len_val_abnormal = val_abnormal_data.shape[0]

    len_test_normal = test_normal_data.shape[0]
    len_test_abnormal = test_abnormal_data.shape[0]

    train_data = np.concatenate([train_normal_data, train_abnormal_data], axis=0)
    train_label = np.concatenate([np.zeros(len_train_normal), np.ones(len_train_abnormal)], axis=0)

    val_data = np.concatenate([val_normal_data, val_abnormal_data], axis=0)
    val_label = np.concatenate([np.zeros(len_val_normal), np.ones(len_val_abnormal)], axis=0)

    test_data = np.concatenate([test_normal_data, test_abnormal_data], axis=0)
    test_label = np.concatenate([np.zeros(len_test_normal), np.ones(len_test_abnormal)], axis=0)

    train_label, train_idx = shuffle_label(train_label, SEED)
    train_data = train_data[train_idx]
    train_label = train_label[train_idx]

    val_label, val_idx = shuffle_label(val_label, SEED)
    val_data = val_data[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_label, SEED)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]

    return train_data, train_label, val_data, val_label, test_data, test_label

def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index