import pickle
import os
import wfdb
import numpy as np
import scipy.interpolate as spi

def down_sample(signal, sample_rate_1, sample_rate_2):   # sample_rate_1 ----> sample_rate_2
    '''
    :param signal:
    :param sample_rate_1:  初始采样率
    :param sample_rate_2:  需要的采样率
    :return:
    '''
    data_ds = []
    for i in range(signal.shape[0]):

        X = np.arange(0, len(signal[i])*(1/sample_rate_1), (1/sample_rate_1)) #4s  步长0.004对应250HZ  1000个点
        new_X = np.arange(0, len(signal[i])*(1/sample_rate_1), (1/sample_rate_2)) # 4S  步长0.002对应500HZ  2000个点

        if X.shape[0] != signal[i].shape[0]:
            X = X[:-1]
        ipo3 = spi.splrep(X, signal[i], k=3)
        iy3 = spi.splev(new_X, ipo3)
        # w = signal[i]
        # print(w[:50])
        data_ds.append(iy3)

    data_ds = np.array(data_ds)

    return data_ds


def load_ref(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    ann_ref = wfdb.rdann(sample_path, 'atr')

    fs = fields['fs']
    length = len(sig)
    sample_descrip = fields['comments']

    beat_loc = np.array(ann_ref.sample)  # r-peak locations
    ann_note = np.array(ann_ref.aux_note)  # rhythm change flag

    af_start_scripts = np.where((ann_note == '(AFIB') | (ann_note == '(AFL'))[0]
    af_end_scripts = np.where(ann_note == '(N')[0]

    if 'non atrial fibrillation' in sample_descrip:
        class_true = 0
    elif 'persistent atrial fibrillation' in sample_descrip:
        class_true = 1
    elif 'paroxysmal atrial fibrillation' in sample_descrip:
        class_true = 2
    else:
        print('Error: the recording is out of range!')

        return -1

    return sig, fs, length, beat_loc, af_start_scripts, af_end_scripts, class_true

def data_process(sample_path):
    # sig, _, fs = load_data(sample_path)
    data = []
    sig, fs, len_sig, beat_loc, af_starts, af_ends, class_true = load_ref(sample_path)
    if class_true == 2:
        print()
    if class_true==0:
        sig = sig.T
        x = down_sample(sig,fs,100)
        # x = sig
        for i in range(1000,x.shape[1],1000):
            data.append(x[:,i-1000:i])
        data = np.asarray(data)
    else:
        sig = sig.T
        for start,end in zip(af_starts,af_ends):
            segment = sig[:,beat_loc[start]:beat_loc[end]]
            x = down_sample(segment, fs, 100)
            # x = segment
            for i in range(1000, x.shape[1], 1000):
                data.append(x[:, i - 1000:i])
        data = np.asarray(data)

    return data,class_true

if __name__=="__main__":
    DATA_PATH = "/home/chenpeng/workspace/Noisy_MultiModal/experiments/datasets/CSPC/training_II/"
    RESULT_PATH = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/"
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        # if sample == "data_66_12":
        #     print()
        # else:
        #     continue
        sample_path = os.path.join(DATA_PATH, sample)
        data,class_true = data_process(sample_path)

        if class_true == 0:
            name = str(sample)+"_"+"normal.pickle"
        else:
            name = str(sample) + "_" + "fangchan.pickle"

        if len(data.shape) == 3:
            save_name = RESULT_PATH + str(name)
            print(save_name)
            with open(save_name, 'wb') as f:
                pickle.dump(data, f)
