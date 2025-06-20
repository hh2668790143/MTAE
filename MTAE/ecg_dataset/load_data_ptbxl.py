import pickle

import numpy as np


def load_ptbxl_data(opt):
    AF_data_path = '/home/hanhan/workspace/My/utils_2023/data_processing/ptbxl_data/AF_data.pickle'
    NAF_data_path = '/home/hanhan/workspace/My/utils_2023/data_processing/ptbxl_data/NORM_data.pickle'

    with open(AF_data_path, 'rb') as f:
        # 使用pickle的load()方法从文件中反序列化对象
        AF_data = pickle.load(f)

    with open(NAF_data_path, 'rb') as f:
        # 使用pickle的load()方法从文件中反序列化对象
        NAF_data = pickle.load(f)

    AF_data = AF_data.transpose(0, 2, 1)
    NAF_data = NAF_data.transpose(0, 2, 1)

    np.random.seed(opt.seed)
    np.random.shuffle(AF_data)
    np.random.shuffle(NAF_data)
    # NAF_data = NAF_data[:10000]
    NAF_data = NAF_data[:6000]

    NAF_data_len = len(NAF_data)
    AF_data_len = len(AF_data)

    train_X = NAF_data[0:NAF_data_len * 2 // 3]

    val_normal_X = NAF_data[NAF_data_len * 2 // 3: NAF_data_len * 5 // 6]
    val_abnormal_X = AF_data[:AF_data_len // 2]

    test_normal_X = NAF_data[NAF_data_len * 5 // 6:]
    test_abnormal_X = AF_data[AF_data_len // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X



# def load_ptbxl_data(opt):
#     AF_data_path = '/home/hanhan/workspace/My/utils_2023/data_processing/ptbxl_data/AF_data.pickle'
#     NAF_data_path = '/home/hanhan/workspace/My/utils_2023/data_processing/ptbxl_data/NORM_data.pickle'
#
#     # AF_data_path = '/repository/dataset/data_hh/ptbxl/AFIB_data.pickle'
#     # NAF_data_path = '/repository/dataset/data_hh/ptbxl/NORM_data.pickle'
#
#     with open(AF_data_path, 'rb') as f:
#         # 使用pickle的load()方法从文件中反序列化对象
#         AF_data = pickle.load(f)
#
#     with open(NAF_data_path, 'rb') as f:
#         # 使用pickle的load()方法从文件中反序列化对象
#         NAF_data = pickle.load(f)
#
#     AF_data = AF_data.transpose(0, 2, 1)
#     NAF_data = NAF_data.transpose(0, 2, 1)
#
#     np.random.seed(opt.seed)
#     np.random.shuffle(AF_data)
#     np.random.shuffle(NAF_data)
#
#
#     NAF_data_len = len(NAF_data)
#     AF_data_len = len(AF_data)
#
#     train_X = NAF_data[0:8000]
#
#     val_normal_X = NAF_data[8000: 8000 + (NAF_data_len - 8000) // 2]
#     val_abnormal_X = AF_data[:AF_data_len // 2]
#
#     test_normal_X = NAF_data[8000 + (NAF_data_len - 8000) // 2:]
#     test_abnormal_X = AF_data[AF_data_len // 2:]
#
#     return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X
