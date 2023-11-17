import time
from torch.optim import Adam
import torch

from utils import AverageMeter


class Model(torch.nn.Module):
    def __init__(self, feature_extractor):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier0 = torch.nn.Sequential(
            torch.nn.Linear(feature_extractor.feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

        self.classifier1 = torch.nn.Sequential(
            torch.nn.Linear(feature_extractor.feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )
        self.classifier2 = torch.nn.Sequential(
            torch.nn.Linear(feature_extractor.feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )

        self.classifier3 = torch.nn.Sequential(
            torch.nn.Linear(feature_extractor.feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )
        self.classifier4 = torch.nn.Sequential(
            torch.nn.Linear(feature_extractor.feature_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 1),
        )


        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam([{"params": self.feature_extractor.parameters(), "lr": 0.1},
                               {"params": self.classifier0.parameters(), "lr": 0.1},
                               {"params": self.classifier1.parameters(), "lr": 0.1},
                               {"params": self.classifier2.parameters(), "lr": 0.1},
                               {"params": self.classifier3.parameters(), "lr": 0.1},
                               {"params": self.classifier4.parameters(), "lr": 0.1},
                               ])



    def forward(self, x):
        out = self.feature_extractor(x)
        c_output0 = self.classifier0(out)
        c_output1 = self.classifier1(out)
        c_output2 = self.classifier2(out)
        c_output3 = self.classifier3(out)
        c_output4 = self.classifier4(out)
        c_output = torch.cat([c_output0, c_output1, c_output2, c_output3, c_output4], dim=1)

        return c_output

    def train(self, epoch, train_loader):
        start_time = time.time()
        self.feature_extractor.train()
        self.classifier0.train()
        self.classifier1.train()
        self.classifier2.train()
        self.classifier3.train()
        self.classifier4.train()

        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        for i, (data, target, _) in enumerate(train_loader):
            # data, target = _data_augmentation(data)
            if torch.cuda.is_available():
                data = torch.cat(data, 0).cuda()
                target = torch.cat(target, 0).cuda()
            self.optimizer.zero_grad()
            output = self.forward(data)
            # print(target)
            loss = self.ce(output, target)
            loss_meter.update(loss.item(), len(target))
            loss.backward()
            self.optimizer.step()
            pred = output.argmax(-1)
            correct = pred.eq(target.view_as(pred)).cpu().sum()
            accuracy = (100.0 * correct / float(len(target)))
            accuracy_meter.update(accuracy.item(), len(target))
        elapsed_time = time.time() - start_time
        print("Epoch [" + str(epoch) + "]"
              + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
              + " loss: " + str(loss_meter.avg)
              + "; acc.: " + str(accuracy_meter.avg))
        return loss_meter.avg, accuracy_meter.avg

    def save(self, file_path="./checkpoint.dat"):
        # state_dict = self.classifier.state_dict()
        feature_extractor_state_dict = self.feature_extractor.state_dict()
        classifier0_state_dict = self.classifier0.state_dict()
        classifier1_state_dict = self.classifier1.state_dict()
        classifier2_state_dict = self.classifier2.state_dict()
        classifier3_state_dict = self.classifier3.state_dict()
        classifier4_state_dict = self.classifier4.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": feature_extractor_state_dict,
                    "classifier0": classifier0_state_dict,
                    "classifier1": classifier1_state_dict,
                    "classifier2": classifier2_state_dict,
                    "classifier3": classifier3_state_dict,
                    "classifier4": classifier4_state_dict,
                    "optimizer": optimizer_state_dict},
                    file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.classifier0.load_state_dict(checkpoint["classifier0"])
        self.classifier1.load_state_dict(checkpoint["classifier1"])
        self.classifier2.load_state_dict(checkpoint["classifier2"])
        self.classifier3.load_state_dict(checkpoint["classifier3"])
        self.classifier4.load_state_dict(checkpoint["classifier4"])
        self.feature_extractor.load_state_dict(checkpoint["backbone"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
