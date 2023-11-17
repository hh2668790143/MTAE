import glob
import os
import pickle

import numpy as np


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

def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index

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


def load_data_cpsc2021(root, len_num=1000, seed=1024):
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
