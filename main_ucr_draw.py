import os
import random

from MlultModal.dataset.UCR_dataloader_noise_cpsc import load_data_cpsc, load_data_icentia11k, load_data_ptbxl
from tool.CPSC_dataloader.load_data_cpsc2021 import load_ucr
from tool.Do_Hist import do_hist
from tool.TSNE import do_tsne_sns, tsne_3D

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
from options import Options
# from dataset.UCR_dataloader import load_data
import numpy as np

device = torch.device("cuda:1" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

DATASETS_NAME = {
    # 'CBF': 3,
    # 'FreezerSmallTrain': 2,
    # 'TwoPatterns': 4,
    # 'RefrigerationDevices': 3,
    # 'FreezerRegularTrain': 2,
    # 'SmallKitchenAppliances': 3,


    # 'cpsc': 1,
    # 'ptbxl': 1,
    # 'icentia11k': 1,
    'WADI': 1,

}
SEEDS = [
    1, 2, 3
    # 0, 1, 2
]

if __name__ == '__main__':

    opt.noisy_classify = 5
    opt.nz = 64
    opt.batchsize = 1280
    opt.Snr = 75
    opt.FM = False
    opt.filter_type = 'no'      # hp  l1
    opt.model = 'AE_CNN_self_88'

    if opt.model == "BeatGAN":
        from model.BeatGAN import BeatGAN as ModelTrainer
    elif opt.model == 'AE_CNN_self_88':
        from model.AE_CNN_self_88 import ModelTrainer
    elif opt.model == 'USAD':
        from model.USAD import UsadModel as ModelTrainer, training_draw

    else:
        raise Exception("no this model_eeg :{}".format(opt.model))

    for dataset_name in list(DATASETS_NAME.keys()):
        for normal_idx in range(DATASETS_NAME[dataset_name]):
            opt.numclass = DATASETS_NAME[dataset_name]
            opt.dataset = dataset_name
            if opt.dataset in ['cpsc']:
                opt.nc = 2
            elif opt.dataset in ['ptbxl']:
                opt.nc = 12
            elif opt.dataset in ['icentia11k']:
                opt.nc = 1

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))

            for seed in SEEDS:
                # Set seed
                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                opt.seed = seed
                opt.normal_idx = normal_idx
                if dataset_name == 'cpsc':
                    dataloader, opt.isize, opt.signal_length = load_data_cpsc(opt)
                elif dataset_name == 'icentia11k':
                    dataloader, opt.isize, opt.signal_length = load_data_icentia11k(opt)
                elif dataset_name == 'ptbxl':
                    dataloader, opt.isize, opt.signal_length = load_data_ptbxl(opt)
                else:
                    dataloader, opt.isize = load_ucr(opt, dataset_name)
                opt.dataset = dataset_name
                print(opt)
                print("################", dataset_name, "##################")

                seed_index = "SEED" + str(seed)

                if opt.model == "BeatGAN":
                    model = ModelTrainer(opt, dataloader, device)
                    m = torch.load(
                        "./hh_UCR_2/{}/{}/{}/model/model_{}.pth".format(opt.dataset, opt.Snr, opt.model, opt.model))
                    model.G.load_state_dict(m[seed_index]["Generator"])
                    model.D.load_state_dict(m[seed_index]["Discriminator"])
                    y_true, y_pred, latent = model.test_draw()
                    do_hist(y_pred, y_true, opt)
                    do_tsne_sns(latent, y_true, opt)
                    # tsne_3D(latent, y_true)

                elif opt.model == "AE_CNN_self_88":
                    model = ModelTrainer(opt, dataloader, device)
                    m = torch.load(
                        "./hh_UCR_1/{}/{}/{}/model/model_{}.pth".format(opt.dataset, opt.Snr, opt.model, opt.model))
                    model.encoder.load_state_dict(m[seed_index]["encoder"])
                    model.decoder.load_state_dict(m[seed_index]["decoder"])
                    model.decoder_f.load_state_dict(m[seed_index]["decoder_f"])
                    y_true, y_pred, latent1, latent2 = model.test_draw()
                    do_hist(y_pred, y_true, opt)
                    # opt.dataset = opt.dataset + '_{}'.format(1)
                    # do_tsne_sns(latent1, y_true, opt)
                    # opt.dataset = opt.dataset + '_{}'.format(2)
                    # do_tsne_sns(latent2, y_true, opt)

                elif opt.model == "USAD":
                    model = ModelTrainer(opt, device)
                    m = torch.load(
                        "./hh_UCR_USAD/{}/model/model_{}.pth".format(opt.dataset, opt.model))
                    model.encoder.load_state_dict(m[seed_index]["encoder"])
                    model.decoder1.load_state_dict(m[seed_index]["decoder1"])
                    model.decoder2.load_state_dict(m[seed_index]["decoder2"])
                    y_true, y_pred, latent1, latent2 = training_draw(opt, 1000, model,
                                                                     dataloader['train'],
                                                                     dataloader['val'],
                                                                     dataloader['test'])
                    do_hist(y_pred, y_true, opt)
                    # opt.dataset = opt.dataset + '_{}'.format(1)
                    y_true = y_true.astype(np.int32)
                    do_tsne_sns(latent1, y_true, opt)
                    # opt.dataset = opt.dataset + '_{}'.format(2)
                    # do_tsne_sns(latent2, y_true, opt)
