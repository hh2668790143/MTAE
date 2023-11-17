# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/11/21 22:53
import os
import torch
from options import Options
import numpy as np
from tool.CPSC.load_data_cpsc2021 import data_2_dataset,load_CPSC2021_data_1,getDataSet

device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
opt = Options().parse()


if opt.model == "BeatGAN":
    from model.BeatGAN import BeatGAN as ModelTrainer
elif opt.model == "AE_CNN":
    from model.AE_CNN import ModelTrainer
elif opt.model == 'AE_LSTM':
    from  model.AE_LSTM import  ModelTrainer
elif opt.model == 'Ganomaly':
    from model.Ganomaly import Ganomaly as ModelTrainer
elif opt.model == 'AE_CNN_noisy_multi':
    from model.AE_CNN_noisy_multi import ModelTrainer
elif opt.model == 'AE_CNN_self':
    from model.AE_CNN_self import ModelTrainer
elif opt.model == 'AE_CNN_self_2':
    from model.AE_CNN_self_2 import ModelTrainer

else:
    raise Exception("no this model_eeg :{}".format(opt.model))

DATASETS_NAME={
    'CSPC':1,
    'CWRU': 1,
    'MFPT': 1,
    'EpilepticSeizure': 1,
    # # # Image:
    'ProximalPhalanxOutlineCorrect': 2,
      'Yoga': 2,
      'Crop': 24,
    'MixedShapesRegularTrain': 5,

    #ECG:
     'ECG5000':2,
     'TwoLeadECG':2,
        'ECGFiveDays':2,
    # #
    # # #SPECTRO:
    # #
    'Strawberry': 2,
      'EthanolLevel': 4,#
    #
     #Sensor:
       'StarLightCurves': 3,

    ###'FordA': 2,


    # #Simulated:
       'TwoPatterns': 4,
       'CBF': 3,
       'Mallat': 8,
    #
    #
     'CinCECGTorso':4,
    #
    #Device:
     'ElectricDevices': 7,
     'LargeKitchenAppliances': 3,
    'Computers':2,

    #Motion
    'UWaveGestureLibraryX':8,
    'UWaveGestureLibraryY':8,
     'UWaveGestureLibraryZ':8,
    #
    # #Traffic:
     'MelbournePedestrian':10,


    # #ADD
     'RefrigerationDevices':3,
     'ChlorineConcentration':3,
    #
    # #'Fish':7,
    #
     'ScreenType':3,
     'SmallKitchenAppliances':3,
     'CinCECGTorso':4,


}
SEEDS=[
    1,2,3,4,5,6
]

if __name__ == '__main__':

    results_dir='./log106'

    opt.outf = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
    file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')

    import datetime
    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
    file2print.flush()
    file2print_detail.flush()

    # dataset_name="CSPC"
    # 噪声种类
    # opt.plt_show=True
    root = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/"
    opt.noisy_classify = 5
    opt.nc = 2

    AUCs={}
    APs={}
    MAX_EPOCHs = {}
    error=1
    for i in range(10):
        opt.Snr=i+15
        MAX_EPOCHs_seed = {}
        AUCs_seed = {}
        APs_seed = {}
        for seed in SEEDS:
            train_data, train_label, test_data, test_label=load_CPSC2021_data_1(root,seed)
            np.random.seed(seed)
            opt.seed = seed
            for i in range(5):
                dataloader, opt.isize=getDataSet(train_data, train_label, test_data, test_label, opt, 5, i)
                model = ModelTrainer(opt, dataloader, device)

            opt.name = "%s/%s" % (opt.model, opt.dataset)
            expr_dir = os.path.join(opt.outf, opt.name, 'train')
            test_dir = os.path.join(opt.outf, opt.name, 'test')

            if not os.path.isdir(expr_dir):
                os.makedirs(expr_dir)
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)

            args = vars(opt)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

            print(opt)
            print("################  Train  ##################")
            ap_test, auc_test, epoch_max_point = model.train()
            print("SEED: {}\t{}\t{}\t{}".format(seed,auc_test,ap_test,epoch_max_point),file=file2print)
            file2print.flush()
            AUCs_seed[seed] = auc_test
            APs_seed[seed] = ap_test
            MAX_EPOCHs_seed[seed] = epoch_max_point
        if error==0:
            continue
        MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
        AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
        AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
        APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
        APs_seed_std = round(np.std(list(APs_seed.values())), 4)

        print("AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={}".format(
            AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed))

        print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
            opt.model,AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
            MAX_EPOCHs_seed_max
        ), file=file2print_detail)
        file2print_detail.flush()





