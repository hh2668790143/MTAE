import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from options import Options

from ecg_dataset.load_data_cpsc2021 import load_data_cpsc2021
from ecg_dataset.load_data_icential11k import load_icentia11k_data
from ecg_dataset.load_data_ptbxl import load_ptbxl_data
from model_utils import transform


def list2hot(label):
    res = []
    for i in ['Gussian', 'Gamma', 'Rayleign', 'Exponential', 'Uniform']:
        if i in label:
            res.append(1)
        else:
            res.append(0)
    return res


def list2hot_f(label):
    res = []
    for i in ['Kalman', 'Wiener']:
        if i in label:
            res.append(1)
        else:
            res.append(0)
    return res


def process_item(i, snr, p):
    gussian = transform.Gussian(snr=snr, p=p)
    gamma = transform.Gamma(snr=snr, p=p)
    rayleign = transform.Rayleign(snr=snr, p=p)
    exponential = transform.Exponential(snr=snr, p=p)
    uniform = transform.Uniform(snr=snr, p=p)
    transforms_list = {
        'gussian': gussian,
        'gamma': gamma,
        'rayleign': rayleign,
        'exponential': exponential,
        'uniform': uniform
    }
    trans = transform.Compose_noise(transforms_list.values(), snr=snr)
    data, fix_noisy, label = trans(i)

    kalman = transform.Kalman(p=p)
    transforms_list_f = {
        'kalman': kalman,
    }
    trans_f = transform.Compose_filter(transforms_list_f.values())
    data_f, fix_noisy_f, label_f = trans_f(i)

    return (data, fix_noisy, list2hot(label), data_f, fix_noisy_f, list2hot_f(label_f))


def FN_process(x, snr=75):
    x_ecg = x
    # 初始化空列表
    fix_data = []  # 混合噪声
    fix_noisy_data = []  # 噪声混合
    labels = []
    fix_data_f = []  # 混合噪声
    fix_noisy_data_f = []  # 噪声混合
    labels_f = []

    # 设置参数
    p = 0.5  # 请设置P值
    n_jobs = 32  # 使用所有可用的核心，或设置为具体数字如4

    # 使用joblib进行并行处理
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(process_item)(i, snr, p) for i in x_ecg
    # )

    # 获取数据长度
    total_items = len(x_ecg)

    # 使用 tqdm 包装 Parallel 迭代器
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_item)(i, snr, p) for i in tqdm(x_ecg, total=total_items, desc="Processing data", miniters=1)
    )


    # 收集结果
    for data, fix_noisy, label, data_f, fix_noisy_f, label_f in results:
        fix_data.append(data)
        fix_noisy_data.append(fix_noisy)
        labels.append(label)
        fix_data_f.append(data_f)
        fix_noisy_data_f.append(fix_noisy_f)
        labels_f.append(label_f)

    return fix_data, fix_noisy_data, labels, fix_data_f, fix_noisy_data_f, labels_f


opt = Options().parse()

if __name__ == '__main__':

    opt.seed = 2
    opt.dataset = 'icentia11k'  # cpsc2021   ptbxl   icentia11k
    opt.section = 'train'  # train    val  test

    if opt.dataset == 'cpsc2021':
        train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
            '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000, opt.seed)

    elif opt.dataset == 'icentia11k':
        train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
            '/data/icentia11k/', opt.seed)
        train_data = np.expand_dims(train_data, 1)
        val_normal_data = np.expand_dims(val_normal_data, 1)
        val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
        test_normal_data = np.expand_dims(test_normal_data, 1)
        test_abnormal_data = np.expand_dims(test_abnormal_data, 1)

    elif opt.dataset == 'ptbxl':
        train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_ptbxl_data(opt)

    x_ecg = train_data

    x_ecg = x_ecg[:50, :, :]

    fix_data, fix_noisy_data, labels, fix_data_f, fix_noisy_data_f, labels_f = FN_process(x_ecg)


    # # 初始化空列表
    # fix_data = []  # 混合噪声
    # fix_noisy_data = []  # 噪声混合
    # labels = []
    # fix_data_f = []  # 混合噪声
    # fix_noisy_data_f = []  # 噪声混合
    # labels_f = []
    #
    # # 设置参数
    # snr = 75  # 请设置SNR值
    # p = 0.5  # 请设置P值
    # n_jobs = 16  # 使用所有可用的核心，或设置为具体数字如4
    #
    # # 使用joblib进行并行处理
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(process_item)(i, snr, p) for i in x_ecg
    # )
    #
    # # 收集结果
    # for data, fix_noisy, label, data_f, fix_noisy_f, label_f in results:
    #     fix_data.append(data)
    #     fix_noisy_data.append(fix_noisy)
    #     labels.append(label)
    #     fix_data_f.append(data_f)
    #     fix_noisy_data_f.append(fix_noisy_f)
    #     labels_f.append(label_f)
    #
    # if not os.path.exists(
    #         "./FN/{}/{}/{}".format('FN', opt.dataset, opt.section)):
    #     os.makedirs("./FM/{}/{}/{}".format('FN', opt.dataset, opt.section))
    #
    # np.save("./FN/{}/{}/{}/{}.npy".format('FN', opt.dataset, opt.section, opt.seed), fix_data)
    #
    # print('FN down')
