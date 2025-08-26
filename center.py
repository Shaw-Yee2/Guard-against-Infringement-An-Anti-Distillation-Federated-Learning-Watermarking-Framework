from torch.nn import Module
from torch.utils.data import DataLoader
from fed_aggregation import FedAggregation
import os
import torch
import datetime
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from copy import deepcopy


class Center:

    def __init__(self,
                 center_name: str,
                 init_model: Module,
                 dataloader: DataLoader,
                 loss_fn: _Loss,
                 optim: Optimizer,
                 client_list: list = None,
                 epoch: int = 100,
                 device: str = "cpu"
                 ):
        self.center_name = center_name
        self.center_model = init_model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optim = optim
        self.client_list = client_list
        self.device = device
        self.client_model_dict = {}
        self.epoch = 0
        self.train_epoch = epoch

        self.center_path = "./centers/" + center_name + "/"
        self.model_path = "./centers/" + center_name + "/models/"
        if not os.path.exists("./centers/"):
            os.mkdir("./centers/")
        if not os.path.exists(self.center_path):
            os.mkdir(self.center_path)

        with open(self.center_path + "log.info", "w") as file:
            print(self.center_name, "build")
            file.write(self.center_path + "\n")
            file.write("==========create" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            for n, _ in self.center_model.named_parameters():
                file.write(str(n) + "\n")

        if self.client_list is not None:
            for client in client_list:
                with open(self.center_path + "log.info", "a+") as file:
                    file.write("==========add" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                    file.write(client.client_name + "\n")
                self.client_model_dict[client.client_name] = client.model
                client.connect(self)
        else:
            self.client_list = []

    def add_client(self, client):
        if client not in self.client_list:
            with open(self.center_path + "log.info", "a+") as file:
                file.write("==========add" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write(client.client_name + "\n")
            self.client_list.append(client)

    def update(self, client_name: str, client_model: Module):
        with open(self.center_path + "log.info", "a+") as file:
            file.write("==========update" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            file.write("update" + client_name + "\n")
        self.client_model_dict[client_name] = client_model

    def assign(self):
        with open(self.center_path + "log.info", "a+") as file:
            file.write("==========assign" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            for client in self.client_list:
                client.assign(self.center_model)
                file.write("assign" + client.client_name + "\n")

    def assign_by_name(self, name):
        with open(self.center_path + "log.info", "a+") as file:
            file.write("==========assign" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            for client in self.client_list:
                if name == client.client_name:
                    client.assign(self.center_model)
                    file.write("assign" + client.client_name + "\n")

    def aggregation(self):
        self.epoch += 1
        fed = FedAggregation(self.client_model_dict)
        fed.fed_avg()
        self.center_model = fed.agg_model
        with open(self.center_path + "log.info", "a+") as file:
            file.write("==========aggregation" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            file.write("aggregation\n")

    def save(self):
        torch.save(self.center_model.state_dict(), self.model_path + self.center_name + str(self.epoch) + ".pth")

    def eval(self, epoch):
        self.center_model.eval()
        accurate = []
        total = []
        with tqdm(self.dataloader, ncols=100, leave=True, position=0) as loader:
            for inputs, outputs in loader:
                predicts = torch.argmax(self.center_model(inputs.to(self.device)), 1)
                accurate.append((predicts == outputs.to(self.device)).sum().float())
                total.append(len(outputs))

                accuracy = sum(accurate) / sum(total)

                loader.set_description("Epoch eval" + str(epoch) + ": ")
                loader.set_postfix(accuracy=accuracy)

            with open(self.center_path + "log.info", "a+") as file:
                file.write("==========eval" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write("eval\n" + "Epoch" + str(epoch) + ": " + str(accuracy) + "\n")

    def train(self, save_epoch: int):
        self.center_model.train()
        with open(self.center_path + "log.info", "a+") as file:
            file.write("==========train" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            with tqdm(range(1, self.train_epoch + 1), ncols=100, leave=True) as turn:
                for i in turn:
                    t_loss = 0.0
                    with tqdm(self.dataloader, ncols=100, leave=False, position=0) as in_turn:
                        for inputs, outputs in in_turn:
                            self.optim.zero_grad()

                            predicts = self.center_model(inputs.to(self.device))
                            loss = self.loss_fn(predicts, outputs.to(self.device))
                            t_loss += loss.data
                            loss.backward()
                            self.optim.step()

                            turn.set_postfix(loss=loss)

                    file.write("epoch" + str(i) + "\t" + str(t_loss / len(self.dataloader)) + "\n")

                    turn.set_description(self.center_name + " total epoch" + str(i))
                    turn.set_postfix(loss=t_loss / len(self.dataloader))

                    if i % save_epoch == 0:
                        self.save()
