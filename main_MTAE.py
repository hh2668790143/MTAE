import os
import random

from dataset.UCR_dataloader_noise_cpsc import load_data_cpsc, load_data_icentia11k, load_data_ptbxl
from tool.CPSC_dataloader.load_data_cpsc2021 import load_ucr

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
from options import Options
# from dataset.UCR_dataloader import load_data
from dataset.UCR_dataloader import load_data
import numpy as np
import datetime

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

DATASETS_NAME = {
    # 'CBF': 3,
    # 'FreezerSmallTrain': 2,
    # 'TwoPatterns': 4,
    # 'RefrigerationDevices': 3,
    # 'FreezerRegularTrain': 2,
    # 'SmallKitchenAppliances': 3,
    # 'SWAT': 1,
    # 'WADI': 1,

    # 'Earthquakes': 1,

    'icentia11k': 1,
    # 'cpsc': 1,
    # 'ptbxl': 1,
}
SEEDS = [
    1, 2, 3
    # 2
]

if __name__ == '__main__':

    opt.noisy_classify = 5
    opt.nc = 1
    # opt.batchsize = 128
    opt.batchsize = 512
    # opt.batchsize = 1280 * 2
    opt.Snr = 50
    opt.niter = 1000
    opt.early_stop = 200
    opt.lr = 0.0001
    # opt.Snr = 50
    # opt.Snr = 100
    opt.sigm = 0.5
    opt.FM = False

    opt.model = 'MTAE'

    if opt.model == 'MTAE':
        from model.MTAE import ModelTrainer  # 滤波重构 加噪分类重构
    elif opt.model == 'wo-Classify':
        from model.wo-Classify import ModelTrainer  # 加噪重构，加滤波重构   wo-Classify
    elif opt.model == 'wo-Noise':
        from model.wo-Noise import ModelTrainer  # 加滤波重构    wo-Noise
    elif opt.model == 'wo-Filter':
        from model.wo-Filter import ModelTrainer  # 加噪分类重构    wo-Filter
    elif opt.model == 'AE_CNN_self_final':
        from model.AE_CNN_self_final import ModelTrainer

    else:
        raise Exception("no this model :{}".format(opt.model))

    for dataset_name in list(DATASETS_NAME.keys()):

        results_dir = './hh_MTAE/{}/{}_{}/{}'.format(dataset_name, opt.Snr, opt.sigm, opt.model)

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
        Recall = {}
        F1 = {}
        MAX_EPOCHs = {}

        for normal_idx in range(DATASETS_NAME[dataset_name]):
            opt.numclass = DATASETS_NAME[dataset_name]

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))
            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            pres_seed = {}
            recall_seed = {}
            f1_seed = {}
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
                opt.dataset = dataset_name

                if dataset_name == 'cpsc':
                    dataloader, opt.isize, opt.signal_length = load_data_cpsc(opt)
                elif dataset_name == 'icentia11k':
                    dataloader, opt.isize, opt.signal_length = load_data_icentia11k(opt)
                elif dataset_name == 'ptbxl':
                    dataloader, opt.isize, opt.signal_length = load_data_ptbxl(opt)
                else:
                    dataloader, opt.isize = load_ucr(opt, dataset_name)



                if opt.dataset in ['cpsc']:
                    opt.nc = 2
                elif opt.dataset in ['ptbxl']:
                    opt.nc = 12
                elif opt.dataset in ['icentia11k']:
                    opt.nc = 1

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

                ap_test, auc_test, epoch_max_point, Pre_test, Recall_test, f1_test = model.train()

                print("SEED:{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(seed, dataset_name, auc_test, ap_test, Pre_test,
                                                                   Recall_test, f1_test, epoch_max_point),
                      file=file2print_detail1)
                file2print_detail1.flush()

                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                pres_seed[seed] = Pre_test
                recall_seed[seed] = Recall_test
                f1_seed[seed] = f1_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)
            pres_seed_mean = round(np.mean(list(pres_seed.values())), 4)
            pres_seed_std = round(np.std(list(pres_seed.values())), 4)
            recall_seed_mean = round(np.mean(list(recall_seed.values())), 4)
            recall_seed_std = round(np.std(list(recall_seed.values())), 4)
            f1_seed_mean = round(np.mean(list(f1_seed.values())), 4)
            f1_seed_std = round(np.std(list(f1_seed.values())), 4)

            print(
                "Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{} \t Pres={}+{} \t Recalls={}+{} \t F1={}+{} "
                "\t MAX_EPOCHs={}".format(
                    dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                    pres_seed_mean, pres_seed_std, recall_seed_mean, recall_seed_std, f1_seed_mean, f1_seed_std,
                    MAX_EPOCHs_seed))

            print("{}\t{}\t{}"
                  "\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}"
                  "\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}"
                  "\t{:.4f}\t{:.4f}"
                  "\t{}".format(
                opt.model, dataset_name, normal_idx,
                AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                pres_seed_mean, pres_seed_std, recall_seed_mean, recall_seed_std,
                f1_seed_mean, f1_seed_std,
                MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            Pres[normal_idx] = pres_seed_mean
            Recall[normal_idx] = recall_seed_mean
            F1[normal_idx] = f1_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\tTest"
              "\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}"
              "\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}"
              "\t{:.4f}\t{:.4f}"
              "\t{}".format(
            opt.model, dataset_name,
            np.mean(list(AUCs.values())), np.std(list(AUCs.values())), np.mean(list(APs.values())),
            np.std(list(APs.values())),
            np.mean(list(Pres.values())), np.std(list(Pres.values())), np.mean(list(Recall.values())),
            np.std(list(Recall.values())),
            np.mean(list(F1.values())), np.std(list(F1.values())),
            np.max(list(MAX_EPOCHs.values()))
        ), file=file2print)

        file2print.flush()
