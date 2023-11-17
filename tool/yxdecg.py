import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
# from datasets.transformer import rescale, paa, r_plot, FFT

# import pywt
import pickle
# import utils.transforms as transforms
from tool.CPSC_dataloader.load_data_cpsc2021 import TransformDataset

def get_filelist(dir):
    log_id_list = []
    datalist = []
    label1_list = []
    label2_list = []

    for home, dirs, files in os.walk(dir):
        for filename in files:

            if "_data.npy" in filename:
                # 文件名列表，包含完整路径
                log_id = filename.split('_data.npy')[0]
                datalist.append(os.path.join(home, filename))
                label1_list.append(os.path.join(home, "{}_label1.npy".format(log_id)))
                label2_list.append(os.path.join(home, "{}_label2.npy".format(log_id)))
                # # 文件名列表，只包含文件名
                log_id_list.append(log_id)
    return log_id_list, datalist, label1_list, label2_list


def one_class_labeling(labels, normal_class:int):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index], index

    def __len__(self):
        return self.X.size(0)


class ECG_Dataset():

    def __init__(self, data_dir, dataset_name, model_name):
        super().__init__()

        data_N_X = []
        data_N_Y = []
        data_A_X = []
        data_A_Y = []

        for i in range(1, 100):

            # print(i)
            # name = 'FangChan'  # ALL2  40
            # name = 'FeiDa'  # ALL2  35
            # name = 'GengSi'  #ALL3 27
            # name = 'ZuZhi'  # ALL4 100
            name = 'ZhengChang'
            # name = 'GuoSu'  # 10

            print('Read {}-th to {}-th {}:  {}'.format(1, 100, name, i))
            # name = 'FeiDa'

            with open('{}/YXDECG_{}_data_{}.pickle'.format('/repository/ECG/ALL2', i * 1000, name), 'rb') as handle1:
                # with open('{}/YXDECG_{}_data_{}.pickle'.format('/repository/ECG/ALL2', i*1000, name), 'rb') as handle1:
                # with open('{}/YXDECG_{}_data_N.pickle'.format('/repository/ECG/ALL/No_FeiDa', i * 1000),'rb') as handle1:

                data_N_X_S = pickle.load(handle1)  # (1000,12,10000)

                # with open('{}/YXDECG_{}_label.pickle'.format(opt.data_ECG, i*1000), 'rb') as handle2:
                # data_N_Y = pickle.load(handle2)
                data_N_Y_S = np.zeros(len(data_N_X_S))

            data_N_X_S = np.array(data_N_X_S)
            data_N_Y_S = np.array(data_N_Y_S)

            if i == 1:

                data_N_X = data_N_X_S
                data_N_Y = data_N_Y_S

            else:

                data_N_X = np.concatenate((data_N_X, data_N_X_S))
                data_N_Y = np.concatenate((data_N_Y, data_N_Y_S))

        for i in range(1, 50):

            # print(i)
            # @name = 'ZhengChang'
            name = 'No_ZhengChang'
            # name = 'No_FeiDa'
            # name = 'No_GuoSu'
            print('Read {}-th to {}-th {}:  {}'.format(1, 100, name, i))

            with open('{}/YXDECG_{}_data_N.pickle'.format('/repository/ECG/ALL/No_ZhengChang', i * 1000),
                      'rb') as handle1:
                # with open('{}/YXDECG_{}_data_ZhengChang.pickle'.format('/repository/ECG/ALL2', i * 1000), 'rb') as handle1:

                data_A_X_S = pickle.load(handle1)

                # with open('{}/YXDECG_{}_label_N.pickle'.format(opt.data_ECG,  i*1000), 'rb') as handle2:
                # data_A_Y = pickle.load(handle2)
                data_A_Y_S = np.ones(len(data_A_X_S))

            data_A_X_S = np.array(data_A_X_S)
            data_A_Y_S = np.array(data_A_Y_S)

            if i == 1:

                data_A_X = data_A_X_S
                data_A_Y = data_A_Y_S

            else:

                data_A_X = np.concatenate((data_A_X, data_A_X_S))
                data_A_Y = np.concatenate((data_A_Y, data_A_Y_S))

        data_N_X = data_N_X[:, :, :5000]
        data_A_X = data_A_X[:, :, :5000]

        # Split normal samples
        n_normal = data_N_X.shape[0]
        train_X = data_N_X[:(int(n_normal * 0.6)), ]
        train_Y = data_N_Y[:(int(n_normal * 0.6)), ]

        val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]
        val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]
        test_N_X = data_N_X[int(n_normal * 0.8):]
        test_N_Y = data_N_Y[int(n_normal * 0.8):]

        val_N_X_len = val_N_X.shape[0]
        test_N_X_len = test_N_X.shape[0]
        data_A_X_len = data_A_X.shape[0]

        # Split abnormal samples
        data_A_X_idx = list(range(data_A_X_len))
        # np.random.shuffle(data_A_X_idx)

        val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
        val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
        test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
        test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

        val_X = np.concatenate((val_N_X, val_A_X))
        val_Y = np.concatenate((val_N_Y, val_A_Y))
        test_X = np.concatenate((test_N_X, test_A_X))
        test_Y = np.concatenate((test_N_Y, test_A_Y))

        dataIndex = list(range(val_X.shape[0]))
        np.random.seed(1024)
        np.random.shuffle(dataIndex)
        val_X = val_X[dataIndex]
        val_Y = val_Y[dataIndex]

        dataIndex = list(range(test_X.shape[0]))
        np.random.seed(1024)
        np.random.shuffle(dataIndex)
        test_X = test_X[dataIndex]
        test_Y = test_Y[dataIndex]

        # Normalize
        # x_train_max = np.max(train_X)
        # x_train_min = np.min(train_X)
        # train_X = 2. * (train_X - x_train_min) / (x_train_max - x_train_min) - 1.
        # # Test is secret
        # val_X = 2. * (val_X - x_train_min) / (x_train_max - x_train_min) - 1.
        # test_X = 2. * (test_X - x_train_min) / (x_train_max - x_train_min) - 1.
        #

        print("[INFO] Train: normal={}".format(train_X.shape), )
        print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

        self.rp_size = None

        if model_name in ['ucr_OSCNN', 'ucr_CNN']:

            self.signal_length = train_X.shape[-1]

            train_X = np.expand_dims(train_X, 1)  # (292,1,140)
            test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
            val_X = np.expand_dims(val_X, 1)

            self.train_set = RawDataset(train_X, train_Y)
            self.val_set = RawDataset(val_X, val_Y)
            self.test_set = RawDataset(test_X, test_Y)

    def loaders(self, batch_size, num_workers):
        train = DataLoader(
            dataset=self.train_set,  # torch TensorDataset format
            batch_size=batch_size,  # mini batch size
            shuffle=True,
            num_workers=num_workers,
            drop_last=False)

        val = DataLoader(
            dataset=self.val_set,  # torch TensorDataset format
            batch_size=batch_size,  # mini batch size
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

        test = DataLoader(
            dataset=self.test_set,  # torch TensorDataset format
            batch_size=batch_size,  # mini batch size
            shuffle=False,
            num_workers=num_workers,
            drop_last=False)

        return train, val, test





class DECG_Noise_Dataset():

    def __init__(self, data_dir, model_name):
        super().__init__()

        data_N_X = []
        data_N_Y = []
        data_A_X = []
        data_A_Y = []


        name_normal = '正常'
        log_id_list, datalist, label1_list, label2_list = get_filelist("{}/{}".format(data_dir, name_normal))
        for i, filepath in enumerate(datalist):
            data_N_X_S = np.load(filepath)
            data_N_Y_S = np.load(label2_list[i])
            if i == 0:
                data_N_X = data_N_X_S
                data_N_Y = data_N_Y_S
            else:
                data_N_X = np.concatenate((data_N_X, data_N_X_S))
                data_N_Y = np.concatenate((data_N_Y, data_N_Y_S))

        name_normal = '干扰信号'
        log_id_list, datalist, label1_list, label2_list = get_filelist("{}/{}".format(data_dir, name_normal))
        for i, filepath in enumerate(datalist):
            data_A_X_S = np.load(filepath)
            data_A_Y_S = np.load(label2_list[i])
            if i == 0:
                data_A_X = data_A_X_S
                data_A_Y = data_A_Y_S
            else:
                data_A_X = np.concatenate((data_A_X, data_A_X_S))
                data_A_Y = np.concatenate((data_A_Y, data_A_Y_S))

        # data_N_X = data_N_X[:, :, :1000]
        # data_A_X = data_A_X[:, :, :1000]

        data_ALL_X = np.concatenate((data_N_X, data_A_X))
        data_ALL_Y = np.concatenate((data_N_Y, data_A_Y))
        dataIndex = list(range(data_ALL_X.shape[0]))
        np.random.seed(1024)
        np.random.shuffle(dataIndex)
        data_ALL_X = data_ALL_X[dataIndex]
        data_ALL_Y = data_ALL_Y[dataIndex]


        idx_1 = list(np.where(data_ALL_Y==1)[0])
        idx_2 = list(np.where(data_ALL_Y==2)[0])

        idx_3 = list(np.where(data_ALL_Y==3)[0])
        idx_4 = list(np.where(data_ALL_Y==4)[0])


        # Set A,B as normal
        idx_N=idx_1
        # Set C,D as abnormal
        idx_A=idx_3+idx_4+idx_2


        data_N_X=data_ALL_X[idx_N]
        data_N_Y=np.zeros(data_N_X.shape[0])
        data_A_X=data_ALL_X[idx_A]
        data_A_Y=np.ones(data_A_X.shape[0])


        # Split normal samples
        n_normal = data_N_X.shape[0]
        n_abnormal = data_A_X.shape[0]

        train_X = data_N_X[:(int(n_normal * 0.4)), ]
        train_Y = data_N_Y[:(int(n_normal * 0.4)), ]

        val_N_X = data_N_X[int(n_normal * 0.4):int(n_normal * 0.7)]
        val_N_Y = data_N_Y[int(n_normal * 0.4):int(n_normal * 0.7)]
        test_N_X = data_N_X[int(n_normal * 0.7):]
        test_N_Y = data_N_Y[int(n_normal * 0.7):]

        # val_N_X_len = val_N_X.shape[0]
        # test_N_X_len = test_N_X.shape[0]
        # data_A_X_len = data_A_X.shape[0]

        # Split abnormal samples
        # data_A_X_idx = list(range(n_abnormal))
        # np.random.shuffle(data_A_X_idx)

        val_A_X = data_A_X[:(int(n_abnormal * 0.5))]
        val_A_Y = data_A_Y[:(int(n_abnormal * 0.5))]
        test_A_X = data_A_X[(int(n_abnormal * 0.5)):]
        test_A_Y = data_A_Y[(int(n_abnormal * 0.5)):]

        val_X = np.concatenate((val_N_X, val_A_X))
        val_Y = np.concatenate((val_N_Y, val_A_Y))
        test_X = np.concatenate((test_N_X, test_A_X))
        test_Y = np.concatenate((test_N_Y, test_A_Y))

        # dataIndex = list(range(val_X.shape[0]))
        # np.random.seed(1024)
        # np.random.shuffle(dataIndex)
        # val_X = val_X[dataIndex]
        # val_Y = val_Y[dataIndex]
        #
        # dataIndex = list(range(test_X.shape[0]))
        # np.random.seed(1024)
        # np.random.shuffle(dataIndex)
        # test_X = test_X[dataIndex]
        # test_Y = test_Y[dataIndex]

        # Normalize
        # x_train_max = np.max(train_X)
        # x_train_min = np.min(train_X)
        # train_X = 2. * (train_X - x_train_min) / (x_train_max - x_train_min) - 1.
        # # Test is secret
        # val_X = 2. * (val_X - x_train_min) / (x_train_max - x_train_min) - 1.
        # test_X = 2. * (test_X - x_train_min) / (x_train_max - x_train_min) - 1.
        #

        print("[INFO] Train: normal={}".format(train_X.shape), )
        print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
        print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

        self.rp_size = None

        if model_name in ['ucr_OSCNN', 'ucr_CNN']:

            self.signal_length = train_X.shape[-1]

            train_X = np.expand_dims(train_X, 1)  # (292,1,140)
            test_X = np.expand_dims(test_X, 1)  # (4500,1,141)
            val_X = np.expand_dims(val_X, 1)

            # train_X = preprocess_signals(train_X)
            # test_X = preprocess_signals(test_X)
            # val_X = preprocess_signals(val_X)

            self.train_set = TransformDataset(train_X, train_Y)
            self.val_set = TransformDataset(val_X, val_Y)
            self.test_set = TransformDataset(test_X, test_Y)

            # self.train_set = RawDataset(train_X, train_Y)
            # self.val_set = RawDataset(val_X, val_Y)
            # self.test_set = RawDataset(test_X, test_Y)

    # def loaders(self, batch_size, num_workers):
    #     train = DataLoader(
    #         dataset=self.train_set,  # torch TensorDataset format
    #         batch_size=batch_size,  # mini batch size
    #         shuffle=True,
    #         num_workers=num_workers,
    #         drop_last=False)
    #
    #     val = DataLoader(
    #         dataset=self.val_set,  # torch TensorDataset format
    #         batch_size=batch_size,  # mini batch size
    #         shuffle=False,
    #         num_workers=num_workers,
    #         drop_last=False)
    #
    #     test = DataLoader(
    #         dataset=self.test_set,  # torch TensorDataset format
    #         batch_size=batch_size,  # mini batch size
    #         shuffle=False,
    #         num_workers=num_workers,
    #         drop_last=False)
    #
    #     return train, val, test
    def loader(self,opt):
        dataloader = {"train": DataLoader(
            dataset=self.train_set,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=True,
            num_workers=int(opt.workers),
            drop_last=True),

            "val": DataLoader(
                dataset=self.val_set,  # torch TensorDataset format
                batch_size=opt.batchsize,  # mini batch size
                shuffle=False,
                num_workers=int(opt.workers),
                drop_last=True),

            "test": DataLoader(
                dataset=self.test_set,  # torch TensorDataset format val_dataset
                batch_size=opt.batchsize,  # mini batch size
                shuffle=False,
                num_workers=int(opt.workers),
                drop_last=True)
        }
        return dataloader,1000


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

# from sklearn import preprocessing

def TSC_data_loader(dataset_path,dataset_name):
    print("[INFO] {}".format(dataset_name))

    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

