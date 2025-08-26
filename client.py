import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import os
import datetime
from tqdm import tqdm
from data_utils import distance
from copy import deepcopy


class Client:
    def __init__(self,
                 client_name: str,
                 dataloader: DataLoader,
                 epoch: int,
                 model: Module,
                 optim: Optimizer,
                 loss_fn: _Loss,
                 connection_list: list,
                 device: str,
                 ):
        self.client_name = client_name
        self.dataloader = dataloader
        self.epoch = epoch
        self.cur_epoch = epoch
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.connection_list = connection_list
        self.device = device

        self.client_path = "./clients/" + client_name + "/"
        self.model_path = "./clients/" + client_name + "/models/"
        self.pic_path = "./clients/" + client_name + "/pics/"
        self.watermark_path = "./clients/" + client_name + "/WMNIST/"

        if not os.path.exists("./clients/"):
            os.mkdir("./clients/")

        if not os.path.exists(self.client_path):
            os.mkdir(self.client_path)

        if not os.path.exists(self.pic_path):
            os.mkdir(self.pic_path)

        with open(self.client_path + "log.info", "w") as file:
            file.write(self.client_name + "\n")
            file.write("==========create" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            print(self.client_name, "build")
            for n, _ in self.model.named_parameters():
                file.write(str(n) + "\n")

    def train(self, save_epoch: int):
        self.model.train()
        with open(self.client_path + "log.info", "a+") as file:
            file.write("==========train" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            with tqdm(range(1, self.epoch + 1), ncols=100, leave=True) as turn:
                for i in turn:
                    t_loss = 0.0
                    with tqdm(self.dataloader, ncols=100, leave=False, position=0) as in_turn:
                        for inputs, outputs in in_turn:
                            self.optim.zero_grad()

                            predicts = self.model(inputs.to(self.device))
                            loss = self.loss_fn(predicts, outputs.to(self.device))
                            t_loss += loss.data
                            loss.backward()
                            self.optim.step()

                            turn.set_postfix(loss=loss)

                    file.write("epoch" + str(i) + "\t" + str(t_loss / len(self.dataloader)) + "\n")

                    turn.set_description(self.client_name + " total epoch" + str(i))
                    turn.set_postfix(loss=t_loss / len(self.dataloader))

                    if i % save_epoch == 0:
                        self.save(i)

    def connect(self, center):
        if center not in self.connection_list:
            with open(self.client_path + "log.info", "a+") as file:
                file.write("==========connect" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write(center.center_name + "\n")
            self.connection_list.append(center)
            center.add_client(self)

    def update(self, centers: list):
        for center in centers:
            if center in self.connection_list:
                with open(self.client_path + "log.info", "a+") as file:
                    file.write(
                        "==========update" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                    file.write("update" + center.center_name + "\n")
                center.update(self.client_name, self.model)

    def assign(self, center_model: Module):
        with open(self.client_path + "log.info", "a+") as file:
            file.write("==========assign" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            file.write("assign" + "\n")
        center_model_dict = {name: param for name, param in center_model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = deepcopy(center_model_dict[name])

    def save(self, epoch: int):
        self.cur_epoch = epoch
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        torch.save(self.model.state_dict(), self.model_path + self.client_name + str(epoch) + ".pth")

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path + self.client_name + str(self.cur_epoch) + ".pth"))

    def prox_train(self, max_epoch: int = 100, miu: float = 1e-4):
        self.model.train()
        with open(self.client_path + "log.info", "a+") as file:
            file.write("==========train" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            min_loss = 1e10
            for i in range(max_epoch):
                print("\n===========================" + self.client_name + "Epoch" + str(self.epoch) + "===========================\n")
                t_loss = 0.0
                source_model = deepcopy(self.model)
                for inputs, outputs in tqdm(self.dataloader):
                    self.optim.zero_grad()

                    predicts = self.model(inputs)
                    loss = self.loss_fn(predicts, outputs) + (miu * distance(source_model, self.model, p=2) * 0.5)
                    t_loss += loss
                    loss.backward()
                    self.optim.step()

                file.write("\nepoch" + str(i) + "\t" + str(t_loss / len(self.dataloader)) + "\n")

                if t_loss / len(self.dataloader) < min_loss or i == max_epoch:
                    min_loss = t_loss / len(self.dataloader)
                    self.save(i)
