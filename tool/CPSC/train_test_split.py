import glob
import pickle
import random
import os
import numpy as np

def find_size(patients, paths):
    size = 0
    for p in patients:
        for x in paths:
            if x.split("/")[-1].split("_")[1] == p:
                size += int(os.stat(x).st_size)
    return size


def data_process_and_save(patients, paths, name, save_path):
    path_list = []
    data = []
    for p in patients:
        for x in paths:
            if x.split("/")[-1].split("_")[1] == p:
                path_list.append(x)
    for p in path_list:
        with open(p, 'rb') as handle1:
            data.extend(pickle.load(handle1))
    data = np.asarray(data)
    for i in range(1000, data.shape[0], 1000):
        # data.append(x[:, i - 1000:i])
        x = data[i-1000:i,:,:]
        save_name = os.path.join(save_path,"data_"+str(int(i/1000))+"_"+name+".pickle")
        with open(save_name, 'wb') as f:
            pickle.dump(x, f)
    x = data[i:data.shape[0], :, :]
    save_name = os.path.join(save_path, "data_" + str(int(i / 1000)+1) + "_" + name + ".pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(x, f)

normal_path = sorted(glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_normal.pickle"))
fangchan_path = sorted(glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_fangchan.pickle"))

data_nomal=[]
data_fangchan=[]
for i in normal_path:
    print(i)
    with open(i, 'rb') as handle1:
        data_nomal.append(pickle.load(handle1))
for i in fangchan_path:
    print(i)
    with open(i, 'rb') as handle1:
        data_fangchan.append(pickle.load(handle1))
data_nomal=np.vstack(data_nomal) #(105077, 2, 1000)
data_fangchan=np.vstack(data_fangchan) #(58787, 2, 1000)

root_path = '/home/chenpeng/workspace/dataset/CSPC2021_fanc'
np.save(root_path+"/cpsc_nomal.npy", data_nomal)
np.save(root_path+"/cpsc_fangchan.npy", data_fangchan)
print()
# with open(save_name, 'wb') as f:
#     pickle.dump(x, f)

# print()
# normal_patient = [x.split("/")[-1].split("_")[1] for x in normal_path]
# fangchan_patient = [x.split("/")[-1].split("_")[1] for x in fangchan_path]
#
# normal_patient = list(set(normal_patient))
# fangchan_patient = list(set(fangchan_patient))
# normal_patient = np.sort(normal_patient).tolist()
# fangchan_patient = np.sort(fangchan_patient).tolist()
#
# dataIndex = list(range(len(normal_patient)))
# np.random.seed(1024)
# np.random.shuffle(dataIndex)
# normal_patient = [normal_patient[i] for i in dataIndex]
#
# dataIndex = list(range(len(fangchan_patient)))
# np.random.seed(1024)
# np.random.shuffle(dataIndex)
# fangchan_patient = [fangchan_patient[i] for i in dataIndex]
#
# za = []
# normal_train = []
# fangchan_train  = []
# normal_test = []
# fangchan_test  = []
#
# for i in normal_patient:
#     for j in fangchan_patient:
#         if i == j:
#             za.append(i)
#             normal_train.append(i)
#             fangchan_train.append(i)
#
# for i in za:
#     normal_patient.remove(i)
#     fangchan_patient.remove(i)
#
# normal_patient_size = []
# fangchan_patient_size = []
# for p in normal_patient:
#     p_size = 0
#     for x in normal_path:
#         if x.split("/")[-1].split("_")[1] == p:
#             p_size += int(os.stat(x).st_size)
#     normal_patient_size.append(p_size)
# for p in fangchan_patient:
#     p_size = 0
#     for x in fangchan_path:
#         if x.split("/")[-1].split("_")[1] == p:
#             p_size += int(os.stat(x).st_size)
#     fangchan_patient_size.append(p_size)
#
# # normal_sort = np.sort(normal_patient_size)[::-1]
# # fangchan_sort = np.sort(fangchan_patient_size)[::-1]
# normal_train_size = find_size(normal_train, normal_path)
# fangchan_train_size = find_size(fangchan_train, fangchan_path)
# all_normal_size = np.sum(normal_patient_size) + normal_train_size
# all_fangchan_size = np.sum(fangchan_patient_size) + fangchan_train_size
# test_best_normal_size = all_normal_size * 0.1
# test_best_fangchan_size = all_fangchan_size * 0.1
#
# test_true_normal_size = 0
# test_true_fangchan_size = 0
# for i in range(len(normal_patient)):
#     test_true_normal_size += normal_patient_size[i]
#     normal_test.append(normal_patient[i])
#     if test_true_normal_size >= test_best_normal_size:
#         break
# for i in range(len(fangchan_patient)):
#     test_true_fangchan_size += fangchan_patient_size[i]
#     fangchan_test.append(fangchan_patient[i])
#     if test_true_fangchan_size >= test_best_fangchan_size:
#         break
#
# for p in normal_patient:
#     if p not in normal_test:
#         normal_train.append(p)
#
# for p in fangchan_patient:
#     if p not in fangchan_test:
#         fangchan_train.append(p)
#
# normal_train_size = find_size(normal_train,normal_path)
# fangchan_train_size = find_size(fangchan_train,fangchan_path)
# normal_test_size = find_size(normal_test,normal_path)
# fangchan_test_size = find_size(fangchan_test,fangchan_path)
# print(normal_train_size)
# print(fangchan_train_size)
# print(normal_test_size)
# print(fangchan_test_size)
#
# data_process_and_save(normal_train,normal_path,"normal","/home/chenpeng/workspace/dataset/CSPC2021_fanc/train/")
# data_process_and_save(normal_test,normal_path,"normal","/home/chenpeng/workspace/dataset/CSPC2021_fanc/test/")
# data_process_and_save(fangchan_train,fangchan_path,"fangchan","/home/chenpeng/workspace/dataset/CSPC2021_fanc/train/")
# data_process_and_save(fangchan_test,fangchan_path,"fangchan","/home/chenpeng/workspace/dataset/CSPC2021_fanc/test/")