import glob
import os
import pickle

import numpy as np



def load_iridia_AF(files,seed,pool):
    data = np.nan
    for file_root in files:
        for ecg_path in glob.glob(os.path.join(file_root, 'data_AF_*.pickle')):
            with open(ecg_path, 'rb') as f:
                ecg_data = pickle.load(f)  # 加载pickle文件
                np.random.seed(seed)
                np.random.shuffle(ecg_data)
                if np.isnan(data).all():
                    data = ecg_data[:1000]
                else:
                    data = np.concatenate((data,ecg_data[:1000]))
    # data = data.reshape(-1, 1, data.shape[-1])
    # seg_filter_SQI_return = pool.map(filter_SQI_MultiProcess,
    #                                  data)
    # data = np.expand_dims(np.array(seg_filter_SQI_return), axis=1)
    # data = data.reshape(-1, 2, data.shape[-1])
    return data

def load_iridia_unAF(files,seed,pool):
    data = np.nan
    for file_root in files:
        for ecg_path in glob.glob(os.path.join(file_root, 'data_unAF_*.pickle')):
            with open(ecg_path, 'rb') as f:
                ecg_data = pickle.load(f)  # 加载pickle文件
                np.random.seed(seed)
                np.random.shuffle(ecg_data)
                if np.isnan(data).all():
                    data = ecg_data[:1000]
                else:
                    data = np.concatenate((data,ecg_data[:1000]))
    # data = data.reshape(-1, 1, data.shape[-1])
    # seg_filter_SQI_return = pool.map(filter_SQI_MultiProcess,
    #                                  data)
    # data = np.expand_dims(np.array(seg_filter_SQI_return), axis=1)
    # data = data.reshape(-1, 2, data.shape[-1])
    return data