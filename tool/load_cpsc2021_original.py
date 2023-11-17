import numpy as np
import wfdb
def get_cpsc2021(root_path, sample_rate, start_point, time_len):
    data_x = []
    data_y = []
    trainset1_file_path = root_path + '/training_I/'
    trainset2_file_path = root_path + '/training_II/'
    # load trainsetI
    for line in open(trainset1_file_path + "RECORDS"):
        total_lead_list = []
        line = line.strip('\n')
        print(line)
        for j in range(0, 2):
            #annotation = wfdb.rdann(trainset1_file_path + line, 'atr')  # 位置信息

            record = wfdb.rdrecord(trainset1_file_path + line, sampfrom=start_point,
                                   sampto=sample_rate * time_len, channels=[j])

            ecg = record.p_signal
            ecg = np.concatenate(ecg.reshape((1, -1), order="F"))
            total_lead_list.append(ecg)
            comment = record.comments[0]
        total_lead = np.asfarray(total_lead_list, dtype=np.float32)
        if comment == 'non atrial fibrillation':
            label = 0
        else:
            label = 1
        label = np.asfarray(label, dtype=np.float64)
        data_x.append(total_lead.T)
        data_y.append(label)
    # load trainsetII
    for line in open(trainset2_file_path + "RECORDS"):
        total_lead_list = []
        line = line.strip('\n')
        print(line)
        for j in range(0, 2):
            record = wfdb.rdrecord(trainset2_file_path + line, sampfrom=start_point,
                                   sampto=sample_rate * time_len, channels=[j])
            ecg = record.p_signal
            ecg = np.concatenate(ecg.reshape((1, -1), order="F"))
            total_lead_list.append(ecg)
            comment = record.comments[0]
        total_lead = np.asfarray(total_lead_list, dtype=np.float32)
        if comment == 'non atrial fibrillation':
            label = 0
        else:
            label = 1
        label = np.asfarray(label, dtype=np.float64)
        data_x.append(total_lead.T)
        data_y.append(label)
    data_x = np.asfarray(data_x, dtype=np.float64)
    data_y = np.asarray(data_y, dtype=np.float64)
    return data_x, data_y

if __name__ == '__main__':
    #trainingI/training_I
    root_path = '/home/chenpeng/workspace/Noisy_MultiModal/experiments/datasets/CSPC'
    sample_rate = 200  # 数据采样率
    start_point = 0  # 数据起点
    time_len = 5   # 截取数据长度 单位 秒
    x,y = get_cpsc2021(root_path, sample_rate, start_point, time_len)
    print(x)
    x=x.swapaxes(1,2)
    print(x)
    np.save(root_path+"/cpsc_data.npy", x)
    np.save(root_path+"/cpsc_label.npy", y)