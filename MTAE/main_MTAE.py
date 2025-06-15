import os
import random

from ecg_dataset.ECG_dataloader import get_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import torch
from options import Options
import numpy as np
import datetime

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

DATASETS_NAME = {
    'cpsc2021': 1,
    # 'ptbxl': 1,
    # 'icentia11k': 1,
}

SEEDS = [
    0, 1, 2
    # 1, 2, 3
    # 0,
    # 1,
    # 2,
    # 3,
]
# conda activate my
# python main_MTAE.py

if __name__ == '__main__':
    opt.lr = 0.001
    opt.batchsize = 128

    opt.is_all_data = True
    opt.isDataProcessed = True
    opt.NT = device
    opt.early_stop = 30

    opt.lam = 0.5
    opt.snr = 20

    opt.nz = 20   # 8  20

    opt.model = 'MTAE'

    if opt.model == 'MTAE':
        from model.model_MTAE import ModelTrainer
        opt.augmentation = 'Filter_Noise'

    else:
        raise Exception("no this model:{}".format(opt.model))

    for dataset_name in list(DATASETS_NAME.keys()):

        results_dir = './result/{}/{}/{}_{}'.format(opt.model, dataset_name, opt.snr, opt.lam)

        opt.outf = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')
        file2print_detail1 = open('{}/results_{}_detail1.log'.format(results_dir, opt.model), 'a+')

        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=file2print)
        print(datetime.datetime.now(), file=file2print_detail)
        print(datetime.datetime.now(), file=file2print_detail1)

        file2print.flush()
        file2print_detail.flush()
        file2print_detail1.flush()

        AUCs = {}
        APs = {}
        Pres = {}
        Recalls = {}
        F1s = {}
        MAX_EPOCHs = {}

        for normal_idx in range(DATASETS_NAME[dataset_name]):
            opt.numclass = DATASETS_NAME[dataset_name]
            opt.dataset = dataset_name

            if opt.dataset in ['cpsc2021']:
                opt.nc = 2
            elif opt.dataset in ['ptbxl']:
                opt.nc = 12
            elif opt.dataset in ['icentia11k']:
                opt.nc = 1
            elif opt.dataset in ['IRIDIA_AF']:
                opt.nc = 2

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))
            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            Pres_seed = {}
            Recalls_seed = {}
            F1s_seed = {}
            model_result = {}

            for seed in SEEDS:

                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                opt.seed = seed
                opt.normal_idx = normal_idx

                dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

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

                print("################", dataset_name, "##################")
                print("################  Train  ##################")

                model = ModelTrainer(opt, dataloader, device)

                ap_test, auc_test, epoch_max_point, pre_test, recall_test, f1_test = model.train()

                print("SEED:{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(seed, dataset_name, auc_test,
                                                                                       ap_test, pre_test, recall_test,
                                                                                       f1_test, epoch_max_point),
                      file=file2print_detail1)
                file2print_detail1.flush()

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                Pres_seed[seed] = pre_test
                Recalls_seed[seed] = recall_test
                F1s_seed[seed] = f1_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)
            Pres_seed_mean = round(np.mean(list(Pres_seed.values())), 4)
            Pres_seed_std = round(np.std(list(Pres_seed.values())), 4)
            Recalls_seed_mean = round(np.mean(list(Recalls_seed.values())), 4)
            Recalls_seed_std = round(np.std(list(Recalls_seed.values())), 4)
            F1s_seed_mean = round(np.mean(list(F1s_seed.values())), 4)
            F1s_seed_std = round(np.std(list(F1s_seed.values())), 4)

            print(
                "Dataset: {} \t Normal Label: {} \t {} \t AUCs={}+{} \t APs={}+{} \t Pres={}+{} \t Recalls={}+{} \t F1s={}+{}"
                "\t MAX_EPOCHs={}".format(
                    dataset_name, normal_idx, opt.augmentation,
                    AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                    Pres_seed_mean, Pres_seed_std, Recalls_seed_mean, Recalls_seed_std, F1s_seed_mean, F1s_seed_std,
                    MAX_EPOCHs_seed))

            print("{}\t{}\t{}\t{}\t{}"
                  "\tAUCs={}+{}\tAPs={}+{}\tPres={}+{}\tRecalls={}+{}\tF1s={}+{}"
                  "\t{}".format(
                opt.model, dataset_name, opt.augmentation, opt.lam, normal_idx,
                AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                Pres_seed_mean, Pres_seed_std, Recalls_seed_mean, Recalls_seed_std, F1s_seed_mean, F1s_seed_std,
                MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            Pres[normal_idx] = Pres_seed_mean
            Recalls[normal_idx] = Recalls_seed_mean
            F1s[normal_idx] = F1s_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\t{}\t{}"
              "\tAUCs={}+{}\tAPs={}+{}\tPres={}+{}\tRecalls={}+{}\tF1s={}+{}"
              "\t{}".format(
            opt.model, dataset_name, opt.augmentation, opt.lam,
            np.mean(list(AUCs.values())), np.std(list(AUCs.values())), np.mean(list(APs.values())),
            np.std(list(APs.values())),
            np.mean(list(Pres.values())), np.std(list(Pres.values())), np.mean(list(Recalls.values())),
            np.std(list(Recalls.values())),
            np.mean(list(F1s.values())), np.std(list(F1s.values())),
            np.max(list(MAX_EPOCHs.values()))
        ), file=file2print)

        file2print.flush()
