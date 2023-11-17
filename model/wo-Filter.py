# -*- coding: utf-8 -*-

# Author : chenpeng
# Time : 2022/11/14 21:33
import time, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.optim import Adam
import torch.nn.functional as F

from options import Options
from model.network import weights_init
# from metric import evaluate
from metric import evaluate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
# from MlultModal.draw_ECG.draw_ecg import plot_sample_2
from sklearn.metrics import accuracy_score, r2_score


dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))


class Encoder(nn.Module):
    def __init__(self, ngpu, opt, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320   (1,320) -> (opt.ndf,160)
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),  # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 160
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),  # 32,64
            nn.BatchNorm1d(opt.ndf * 2),  # 归一化处理 参数为需要归一化的维度
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 80
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  # 64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 40

            # nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(opt.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # state size. (ndf*8) x 20
            # nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm1d(opt.ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(opt.ndf * 4, out_z, 2, 1, 0, bias=False),  # 128,50
            # nn.Conv1d(opt.ndf * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
            nn.AdaptiveAvgPool1d(1)  # 平均池化，将最低维度数转化为1个数
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),
            # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, opt.isize),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        # 分类器
        nz = 8
        self.classifier0 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(nz, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

        self.classifier1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(nz, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )
        self.classifier2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(nz, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

        self.classifier3 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(nz, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )
        self.classifier4 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(nz, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, x):
        c_output0 = self.classifier0(x)
        c_output1 = self.classifier1(x)
        c_output2 = self.classifier2(x)
        c_output3 = self.classifier3(x)
        c_output4 = self.classifier4(x)
        c_output = torch.cat([c_output0, c_output1, c_output2, c_output3, c_output4], dim=1)

        return c_output


class ModelTrainer(nn.Module):
    def __init__(self, opt, dataloader, device):
        super(ModelTrainer, self).__init__()
        self.niter = opt.niter  # 训练次数 #1000
        self.dataset = opt.dataset  # 数据集
        self.model = opt.model
        self.outf = opt.outf  # 输出文件夹路径

        self.dataloader = dataloader
        self.device = device
        self.opt = opt

        self.all_loss = []
        self.rec_loss = []
        self.cls_loss = []
        # 每epoch loss
        self.all_loss_epoch = []
        self.rec_loss_epoch = []
        self.cls_loss_epoch = []

        self.batchsize = opt.batchsize  # input batch size 32
        self.nz = opt.nz  # 潜在z向量大小 8
        self.niter = opt.niter

        # self.G = WaveGANGenerator(list_upsample=opt.list_upsample, num_channels=opt.nc, ngpus=opt.ngpu,
        #                           upsample=True).to(device)

        self.encoder = Encoder(opt.ngpu, opt, opt.nz).to(device)
        self.decoder = Decoder(opt.ngpu, opt).to(device)
        self.decoder_f = Decoder(opt.ngpu, opt).to(device)
        self.classifier = classifier().to(device)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

        self.bc_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.sigm = opt.sigm
        # self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerEncoder = optim.Adam([{"params": self.encoder.parameters()},
                                            {"params": self.decoder.parameters()},
                                            {"params": self.classifier.parameters()}],
                                           lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDecoder = optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerDecoder_f = optim.Adam(self.decoder_f.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerClassifier = optim.Adam(self.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizerG, 30, gamma=0.1, last_epoch=-1)

        self.total_steps = 0
        self.cur_epoch = 0
        # 原始的信号
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        # 噪声
        self.input_noise = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                       device=self.device)
        # 混合噪声
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                       device=self.device)
        # 噪声
        self.input_noise_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                         device=self.device)
        # 混合噪声
        self.fixed_input_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                         device=self.device)
        # 原始标签
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)
        # 噪声标签
        self.nosie_label = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        # 噪声标签
        self.nosie_label_f = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)

        self.latent_i_raw = None
        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train AE-CNN-self-66.")
        start_time = time.time()
        best_result = 0
        best_ap = 0
        best_auc = 0
        best_auc_epoch = 0

        early_stop_epoch = 0
        early_stop_auc = 0
        acc_test = []
        acc_val = []
        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch += 1

                # Train
                self.train_epoch()

                self.all_loss_epoch.append(np.sum(self.all_loss) / len(self.all_loss))
                self.rec_loss_epoch.append(np.sum(self.rec_loss) / len(self.rec_loss))
                self.cls_loss_epoch.append(np.sum(self.cls_loss) / len(self.cls_loss))
                self.all_loss = []
                self.rec_loss = []
                self.cls_loss = []

                # Val
                ap, auc, Pre, Recall, f1, acc = self.validate()
                acc_val = acc
                print("acc:" + str(acc_val))
                if auc > best_result:
                    best_result = auc
                    best_auc = auc
                    best_ap = ap
                    best_auc_epoch = self.cur_epoch

                    self.save_model()

                    # Test
                    ap_test, auc_test, Pre_test, Recall_test, f1_test, acc = self.test()
                    acc_test = acc

                    if epoch == 1:
                        early_stop_auc = auc_test

                if auc_test <= early_stop_auc:
                    early_stop_epoch = early_stop_epoch + 1
                else:
                    early_stop_epoch = 0
                    early_stop_auc = auc_test

                if early_stop_epoch == self.opt.early_stop:
                    break

                f.write(
                    "EPOCH [{}] auc:{:.4f} \t  ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print(
                    "EPOCH [{}]  loss:{:.4f} \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, self.err_g, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print("val: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre, Recall, f1))
                print("test: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre_test, Recall_test, f1_test))
                # self.scheduler.step()
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))
        with open(os.path.join(self.outf, self.model, self.dataset, "acc.txt"), "a+") as f:
            f.write("\nval_acc\n")
            f.write(str(acc_val))
            f.write("\ntest_acc\n")
            f.write(str(acc_test))
        with open(os.path.join(self.outf, self.model, self.dataset, "loss.txt"), "a+") as f:
            f.write("\nall_loss\n")
            f.write(str(self.all_loss))
            f.write("\nrec_loss\n")
            f.write(str(self.rec_loss))
            f.write("\ncls_loss\n")
            f.write(str(self.cls_loss))
        # draw_loss(self.all_loss_epoch,
        #           os.path.join(self.outf, self.model, self.dataset, "all_loss_{}.png".format(self.opt.seed)))
        # draw_loss(self.rec_loss_epoch,
        #           os.path.join(self.outf, self.model, self.dataset, "rec_loss_{}.png".format(self.opt.seed)))
        # draw_loss(self.cls_loss_epoch,
        #           os.path.join(self.outf, self.model, self.dataset, "cls_loss_{}.png".format(self.opt.seed)))
        return ap_test, auc_test, best_auc_epoch, Pre_test, Recall_test, f1_test

    def train_epoch(self):

        epoch_start_time = time.time()
        self.encoder.train()
        self.decoder.train()
        self.classifier.train()
        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()
            errors = self.get_errors()
            self.train_hist['G_loss'].append(errors["err_g"])
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

    # def set_input(self, input):
    #     with torch.no_grad():
    #         # input (原始信号0，混合信号1，标签2，噪声3，噪声标签4)
    #         # 原始信号
    #         self.input.resize_(input[0].size()).copy_(input[0])
    #         # 噪声
    #         self.input_noise.resize_(input[3].size()).copy_(input[3])
    #         # 混合噪声
    #         self.fixed_input.resize_(input[1].size()).copy_(input[1])
    #         # 原始标签
    #         self.label.resize_(input[2].size()).copy_(input[2])
    #         # 噪声标签
    #         self.nosie_label.resize_(input[4].size()).copy_(input[4])
    #
    #         # self.nosie_label=np.argmax(self.nosie_label.cpu(), axis=1).to(self.device)
    #         # one_hot = torch.zeros(32, 5).long().to(self.device)
    #         # self.nosie_label=one_hot.scatter_(dim=1, index=self.nosie_label.unsqueeze(dim=1), src=torch.ones(32, 5).long().to(self.device))
    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            # 噪声
            self.input_noise.resize_(input[3].size()).copy_(input[3])
            # 混合噪声
            self.fixed_input.resize_(input[1].size()).copy_(input[1])
            # 原始标签
            self.label.resize_(input[2].size()).copy_(input[2])
            # 噪声标签
            self.nosie_label.resize_(input[4].size()).copy_(input[4])
            # 噪声
            self.input_noise_f.resize_(input[6].size()).copy_(input[6])
            # 混合噪声
            self.fixed_input_f.resize_(input[5].size()).copy_(input[5])
            # 噪声标签
            self.nosie_label_f.resize_(input[7].size()).copy_(input[7])
    def optimize(self):
        self.update_netg()

    def DTWLoss(self, x, y):
        return fastdtw(x, y)

    def update_netg(self):
        self.optimizerEncoder.zero_grad()
        # self.optimizerDecoder.zero_grad()
        # self.optimizerClassifier.zero_grad()
        # 信号，噪声--> rec信号 , , 噪声类别

        output = self.encoder(self.fixed_input)
        output_n = self.encoder(self.input_noise)
        self.rec_singal = self.decoder(output)
        self.noise_classify = self.classifier(output_n)

        self.err_g_rec_signal = self.mse_criterion(self.input, self.rec_singal)  # constrain x' to look like x

        cls1, cls2, cls3, cls4, cls5 = torch.chunk(self.noise_classify, 5, dim=1)
        lb1, lb2, lb3, lb4, lb5 = torch.chunk(self.nosie_label, 5, dim=1)

        # self.err_g_classify = self.bc_criterion(self.noise_classify, self.nosie_label)
        self.err_g_classify = self.BCELoss(cls1, lb1.float()) + self.BCELoss(cls2, lb2.float()) + self.BCELoss(cls3,
                                                                                                               lb3.float()) + self.BCELoss(
            cls4, lb4.float()) + self.BCELoss(cls5, lb5.float())

        self.err_g = self.err_g_rec_signal + self.err_g_classify * self.sigm

        self.all_loss.append(self.err_g.item())
        self.rec_loss.append(self.err_g_rec_signal.item())
        self.cls_loss.append(self.err_g_classify.item())

        self.err_g.backward()
        self.optimizerEncoder.step()
        # self.optimizerDecoder.step()
        # self.optimizerClassifier.step()

    def get_errors(self):

        errors = {
            'err_g': self.err_g.item(),
            'err_g_rec_signal': self.err_g_rec_signal.item(),
            # 'err_g_rec_signal': self.err_g_rec_signal,
            'err_g_classify': self.err_g_classify.item(),
        }
        return errors

    def test(self):
        '''
        test by auc value
        :return: auc
        '''
        y_true, y_pred, acc = self.predict(self.dataloader["test"])
        auc_prc, roc_auc, Pre, Recall, f1 = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1, acc

    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_true, y_pred, acc = self.predict(self.dataloader["val"])
        auc_prc, roc_auc, Pre, Recall, f1 = evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1, acc

    def predict(self, dataloader, scale=False):
        with torch.no_grad():
            # 异常评分
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            # 标签
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            # 噪声error
            # 噪声标签
            self.ns_label = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            # 分类器标签
            self.label1 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            self.label2 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            self.label3 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            self.label4 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            self.label5 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)
            # 预测
            self.cls1_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.cls2_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.cls3_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.cls4_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.cls5_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader, 0):
                self.set_input(data)
                out = self.encoder(self.fixed_input)
                out_n = self.encoder(self.input_noise)
                self.rec_singal = self.decoder(out)
                self.noise_classify = self.classifier(out_n)
                # self.rec_singal, self.noise_classify = self.G(self.input, self.input_noise)
                # 原始和重构均方差
                error_raw = torch.mean(
                    torch.pow(
                        (self.input.view(self.input.shape[0], -1) - self.rec_singal.view(self.rec_singal.shape[0], -1)),
                        2),
                    dim=1)

                cls1, cls2, cls3, cls4, cls5 = torch.chunk(self.noise_classify, 5, dim=1)
                lb1, lb2, lb3, lb4, lb5 = torch.chunk(self.nosie_label, 5, dim=1)
                self.cls1_scores[(i * self.opt.batchsize):(i * self.opt.batchsize + cls1.size(0))] = cls1.reshape(
                    cls1.size(0))
                self.label1[(i * self.opt.batchsize):(i * self.opt.batchsize + lb1.size(0))] = cls1.reshape(lb1.size(0))

                self.cls2_scores[(i * self.opt.batchsize):(i * self.opt.batchsize + cls2.size(0))] = cls2.reshape(
                    cls2.size(0))
                self.label2[(i * self.opt.batchsize):(i * self.opt.batchsize + lb2.size(0))] = cls2.reshape(lb2.size(0))

                self.cls3_scores[(i * self.opt.batchsize):(i * self.opt.batchsize + cls3.size(0))] = cls3.reshape(
                    cls3.size(0))
                self.label3[(i * self.opt.batchsize):(i * self.opt.batchsize + lb3.size(0))] = cls3.reshape(lb3.size(0))

                self.cls4_scores[(i * self.opt.batchsize):(i * self.opt.batchsize + cls4.size(0))] = cls4.reshape(
                    cls4.size(0))
                self.label4[(i * self.opt.batchsize):(i * self.opt.batchsize + lb4.size(0))] = cls4.reshape(lb4.size(0))

                self.cls5_scores[(i * self.opt.batchsize):(i * self.opt.batchsize + cls5.size(0))] = cls5.reshape(
                    cls5.size(0))
                self.label5[(i * self.opt.batchsize):(i * self.opt.batchsize + lb5.size(0))] = cls5.reshape(lb5.size(0))

                self.an_scores[
                (i * self.opt.batchsize):(i * self.opt.batchsize + error_raw.size(0))] = error_raw.reshape(
                    error_raw.size(0))
                # 时序标签
                self.gt_labels[
                (i * self.opt.batchsize):(i * self.opt.batchsize + error_raw.size(0))] = self.label.reshape(
                    error_raw.size(0))

                # 噪声
                # 噪声标签
                # self.ns_label[(i * self.opt.batchsize):(i * self.opt.batchsize + error.size(0))] = self.nosie_label.reshape(
                #     error.size(0))

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()

            acc0 = r2_score(self.label1.cpu().numpy(), self.cls1_scores.cpu().numpy())
            acc1 = r2_score(self.label2.cpu().numpy(), self.cls2_scores.cpu().numpy())
            acc2 = r2_score(self.label3.cpu().numpy(), self.cls3_scores.cpu().numpy())
            acc3 = r2_score(self.label4.cpu().numpy(), self.cls4_scores.cpu().numpy())
            acc4 = r2_score(self.label5.cpu().numpy(), self.cls5_scores.cpu().numpy())
            acc = [acc0, acc1, acc2, acc3, acc4]

            return y_true, y_pred, acc

    def test_draw(self):
        y_true, y_pred, acc = self.predict(self.dataloader["test"])
        return y_true, y_pred, acc, acc



    def save_model(self):
        model_result = {}
        model_result['encoder'] = self.encoder.state_dict()
        model_result['decoder'] = self.decoder.state_dict()
        model_result['classifier'] = self.classifier.state_dict()

        chk_dir = '{}/{}'.format(self.opt.outf, "model")
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)

        torch.save(model_result, chk_dir + '/model_' + self.opt.model + '_' + str(self.opt.seed) + '.pth')

        print('=================================  save model: {}  ================================='.format(
            self.opt.model))




if __name__ == '__main__':
    opt = Options().parse()
    opt.nc = 2
    opt.ndf = 32
    opt.ngpu = 1
    opt.list_upsample = [2, 5]
    noisy_classify = 6
    model = WaveGANGenerator(list_upsample=opt.list_upsample, num_channels=2, ngpus=opt.ngpu, upsample=True)
    print(model.parameters())
    print(model)

    x = torch.Tensor(np.arange(2000 * 32).reshape((32, 2, 1000)))
    y = torch.Tensor(np.arange(2000 * 32).reshape((32, 2, 1000)))
    a, d = model(x, y)

    print()
