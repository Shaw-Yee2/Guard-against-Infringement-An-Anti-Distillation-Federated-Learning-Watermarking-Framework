import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import os
import datetime
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import distance
from copy import deepcopy


class Attacker:
    def __init__(self,
                 attacker_name: str,
                 dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 epoch: int,
                 teacher_model: Module,
                 model: Module,
                 optim: Optimizer,
                 hard_loss_fn: _Loss,
                 soft_loss_fn: _Loss,
                 device: str,
                 ):
        self.attacker_name = attacker_name
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.epoch = epoch
        self.cur_epoch = epoch
        self.teacher_model = teacher_model
        self.model = model
        self.optim = optim
        self.hard_loss_fn = hard_loss_fn
        self.soft_loss_fn = soft_loss_fn
        self.device = device

        self.attacker_path = "./attackers/" + attacker_name + "/"
        self.model_path = "./attackers/" + attacker_name + "/models/"
        self.pic_path = "./attackers/" + attacker_name + "/pics/"
        self.watermark_path = "./attackers/" + attacker_name + "/WMNIST/"

        if not os.path.exists("./attackers/"):
            os.mkdir("./attackers/")

        if not os.path.exists(self.attacker_path):
            os.mkdir(self.attacker_path)

        if not os.path.exists(self.pic_path):
            os.mkdir(self.pic_path)

        with open(self.attacker_path + "log.info", "w") as file:
            file.write(self.attacker_name + "\n")
            file.write("==========create" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            print(self.attacker_name, "build")
            for n, _ in self.model.named_parameters():
                file.write(str(n) + "\n")

    def attack(self, save_epoch: int = 20, temper: int = 7, alpha: float = 0.7):
        self.teacher_model.eval()
        self.model.train()

        with open(self.attacker_path + "log.info", "a+") as file:
            file.write("==========attack" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            with tqdm(range(1, self.epoch + 1), ncols=200, leave=True) as turn:
                for i in turn:
                    t_loss = 0.0
                    h_loss = 0.0
                    s_loss = 0.0
                    for inputs, outputs in self.dataloader:
                        self.optim.zero_grad()

                        with torch.no_grad():
                            teacher_predicts = self.teacher_model(inputs.to(self.device))
                        predicts = self.model(inputs.to(self.device))
                        student_loss = self.hard_loss_fn(predicts, outputs.to(self.device))

                        distillation_loss = self.soft_loss_fn(
                            F.log_softmax(predicts / temper, dim=1),
                            F.softmax(teacher_predicts / temper, dim=1)
                        )

                        loss = alpha * student_loss + (1 - alpha) * distillation_loss * temper * temper

                        t_loss += loss.data
                        h_loss += student_loss.data
                        s_loss += distillation_loss.data
                        loss.backward()
                        self.optim.step()

                        turn.set_postfix(hloss=student_loss, sloss=distillation_loss)

                    file.write("epoch" + str(i) + "\t" + str(t_loss / len(self.dataloader)) + "\n")

                    turn.set_description(self.attacker_name + " total epoch" + str(i))
                    turn.set_postfix(hloss=h_loss / len(self.dataloader), s_loss=s_loss / len(self.dataloader))

                    if i % save_epoch == 0:
                        self.save(i)

    def save(self, epoch: int):
        self.cur_epoch = epoch
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        torch.save(self.model.state_dict(), self.model_path + self.attacker_name + str(epoch) + ".pth")

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path + self.attacker_name + str(self.cur_epoch) + ".pth"))

    def eval(self, epoch):
        self.model.eval()
        accurate = []
        total = []
        with tqdm(self.eval_dataloader, ncols=100, leave=True, position=0) as loader:
            for inputs, outputs in loader:
                predicts = torch.argmax(self.model(inputs.to(self.device)), 1)
                accurate.append((predicts == outputs.to(self.device)).sum().float())
                total.append(len(outputs))

                accuracy = sum(accurate) / sum(total)

                loader.set_description("Epoch eval" + str(epoch) + ": ")
                loader.set_postfix(accuracy=accuracy)

            with open(self.attacker_path + "log.info", "a+") as file:
                file.write("==========attack-eval" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write("eval\n" + "Epoch" + str(epoch) + ": " + str(accuracy) + "\n")
