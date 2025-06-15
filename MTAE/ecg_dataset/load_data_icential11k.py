import os

import numpy as np


def load_icentia11k_data(data_dir, seed):
    # 获取文件夹下所有npy文件的路径
    data_dirs_afib = [os.path.join(data_dir + "afib", f) for f in os.listdir(data_dir + "afib") if f.endswith(".npy")]
    data_dirs_aflut = [os.path.join(data_dir + "aflut", f) for f in os.listdir(data_dir + "aflut") if
                       f.endswith(".npy")]
    data_dirs_normal = [os.path.join(data_dir + "normal", f) for f in os.listdir(data_dir + "normal") if
                        f.endswith(".npy")]

    random_files_afib = data_dirs_afib[1000:1001]
    random_files_normal = data_dirs_normal[1000:1003]
    random_files_normal.append(data_dirs_aflut[4])
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:1]
    # random_files_normal = data_dirs_normal[:3]
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:2]
    # random_files_normal =np.concatenate([data_dirs_normal[3:4],data_dirs_aflut[:4]])
    np.random.seed(seed)
    np.random.shuffle(random_files_afib)
    np.random.shuffle(random_files_normal)
    print(random_files_afib)
    print(random_files_normal)

    # random_files_afib = data_dirs_afib[2000:2001]
    # random_files_normal = data_dirs_normal[1000:1003]

    data_X_afib = np.concatenate([np.load(file) for file in random_files_afib])
    data_X_normal = np.concatenate([np.load(file) for file in random_files_normal])

    np.random.shuffle(data_X_afib)
    np.random.shuffle(data_X_normal)

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_afib[0:len(data_X_afib) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_afib[len(data_X_afib) // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X



def load_icentia11k_data_cls(data_dir, seed):
    # 获取文件夹下所有npy文件的路径
    data_dirs_afib = [os.path.join(data_dir + "afib", f) for f in os.listdir(data_dir + "afib") if f.endswith(".npy")]
    data_dirs_aflut = [os.path.join(data_dir + "aflut", f) for f in os.listdir(data_dir + "aflut") if
                       f.endswith(".npy")]
    data_dirs_normal = [os.path.join(data_dir + "normal", f) for f in os.listdir(data_dir + "normal") if
                        f.endswith(".npy")]

    random_files_afib = data_dirs_afib[1000:1001]
    random_files_normal = data_dirs_normal[1000:1003]
    random_files_normal.append(data_dirs_aflut[4])
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:1]
    # random_files_normal = data_dirs_normal[:3]
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:2]
    # random_files_normal =np.concatenate([data_dirs_normal[3:4],data_dirs_aflut[:4]])
    np.random.seed(seed)
    np.random.shuffle(random_files_afib)
    np.random.shuffle(random_files_normal)
    print(random_files_afib)
    print(random_files_normal)

    # random_files_afib = data_dirs_afib[2000:2001]
    # random_files_normal = data_dirs_normal[1000:1003]

    data_X_afib = np.concatenate([np.load(file) for file in random_files_afib])
    data_X_normal = np.concatenate([np.load(file) for file in random_files_normal])

    np.random.shuffle(data_X_afib)
    np.random.shuffle(data_X_normal)

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_afib[0:len(data_X_afib) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_afib[len(data_X_afib) // 2:]


    train_files_afib = data_dirs_afib[1:2]
    data_afib = np.concatenate([np.load(file) for file in train_files_afib])
    train_abnormal_X = data_afib[0:len(data_afib)]
    train_abnormal_X = train_abnormal_X[0:100]

    return train_X, train_abnormal_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X

