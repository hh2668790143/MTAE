# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2023/4/12 10:28
import numpy as np
import torch
from MlultModal.options import Options
import torch.nn.functional as F
import torch.nn as nn
from MlultModal.tool.CPSC_dataloader.load_data_cpsc2021 import load_data, data_2_dataset, load_icentia11k_data
from tool.TSNE import do_tsne_sns

device = torch.device("cuda:3" if
                      torch.cuda.is_available() else "cpu")
# from MlultModal.model.AE_CNN_self_3 import ModelTrainer
# from MlultModal.model.BeatGAN import BeatGAN as ModelTrainer
# from MlultModal.model.Ganomaly import Ganomaly as ModelTrainer
from MlultModal.model.AE_CNN import ModelTrainer

opt = Options().parse()

opt.model = "Ganomaly"

if opt.model == "BeatGAN":
    from MlultModal.model.BeatGAN import BeatGAN as ModelTrainer
elif opt.model == "AE_CNN":
    from MlultModal.model.AE_CNN import ModelTrainer
elif opt.model == 'Ganomaly':
    from MlultModal.model.Ganomaly import Ganomaly as ModelTrainer
elif opt.model == 'AE_CNN_self_3':
    from MlultModal.model.AE_CNN_self_3 import ModelTrainer
elif opt.model == 'AE_CNN_self_4':
    from MlultModal.model.AE_CNN_self_4 import ModelTrainer
else:
    raise Exception("no this model_eeg :{}".format(opt.model))

if __name__ == '__main__':

    opt.nc = 1
    opt.ndf = 32
    opt.ngpu = 1
    opt.batchsize = 128
    opt.list_upsample = [2, 5]
    noisy_classify = 5
    opt.snr = 1000
    # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
    #     '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)
    # # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
    # #     '/data2/icentia11k/')
    # # train_data = np.expand_dims(train_data, axis=1)
    # # val_normal_data = np.expand_dims(val_normal_data, axis=1)
    # # val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    # # test_normal_data = np.expand_dims(test_normal_data, axis=1)
    # # test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)
    # print('数据读取完成')
    # dataloader, opt.isize = data_2_dataset(train_data, val_normal_data, val_abnormal_data, test_normal_data,
    #                                        test_abnormal_data, opt)
    # model=ModelTrainer(opt,dataloader,device=device)
    # m=torch.load("../model_result1000/model/model_AE_CNN_self_3.pth")
    m1 = torch.load("../11ktestsseed1231000/model/model_Ganomaly.pth")
    # m1=torch.load("../11ktestsseed123snr1000/model/model_AE_CNN_self_4.pth")
    seed = 1
    while True:
        for i in ['SEED1', 'SEED2', 'SEED3', 'SEED4', 'SEED5', 'SEED6']:
            # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
            #     '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000,seed)
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
                '/data2/icentia11k/', seed)
            train_data = np.expand_dims(train_data, axis=1)
            val_normal_data = np.expand_dims(val_normal_data, axis=1)
            val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
            test_normal_data = np.expand_dims(test_normal_data, axis=1)
            test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

            dataloader, opt.isize = data_2_dataset(train_data, val_normal_data, val_abnormal_data, test_normal_data,
                                                   test_abnormal_data, opt)
            print('数据读取完成')
            model = ModelTrainer(opt, dataloader, device=device)
            # model.G.load_state_dict(m1[i])
            model.netg.load_state_dict(m1[i]["Generator"])
            model.netd.load_state_dict(m1[i]["Discriminator"])
            # model.G.load_state_dict(m1[i]["Generator"])
            # model.D.load_state_dict(m1[i]["Discriminator"])

            y_true, y_pred, latent = model.test_draw()
            # y_true, y_pred = model.test_draw()

            # x_train_max = np.max(y_pred)
            # x_train_min = np.min(y_pred)
            # test_X = (y_pred - x_train_min) / (x_train_max - x_train_min)

            # do_hist(y_pred,y_true)
            # do_tsne(latent,y_true)
            do_tsne_sns(latent, y_true)
            # tsne_3D(latent,y_true)
            print()
            seed += 1
        if seed > 20:
            break
    print()
    # model=model.cuda(device=device)
    # y_true,y_pred = model.test_draw()
