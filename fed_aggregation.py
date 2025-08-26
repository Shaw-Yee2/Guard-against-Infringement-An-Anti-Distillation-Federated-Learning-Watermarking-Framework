import torch
from copy import deepcopy


class FedAggregation:
    def __init__(self, model_dict: dict):
        self.model_dict = model_dict
        self.model_num = len(model_dict)
        self.agg_model = deepcopy(list(model_dict.values())[0])
        self.agg_model.zero_grad()

    def show_params(self):
        for name, param in self.agg_model.named_parameters():
            print(name, "\n", param, "\n")

    def fed_avg(self):
        param_sum = {n: torch.zeros_like(p.data) for n, p in self.agg_model.named_parameters()}
        for user in self.model_dict.keys():
            for name, param in self.model_dict[user].named_parameters():
                param_sum[name] += torch.div(param.data, self.model_num)

        for name, param in self.agg_model.named_parameters():
            param.data = param_sum[name]

    def fed_weight_avg(self, weight_list: list):
        if len(weight_list) != self.model_num:
            return
        weight = iter(weight_list)
        param_sum = {n: torch.zeros_like(p.data) for n, p in self.agg_model.named_parameters()}
        for user in self.model_dict.keys():
            model_weight = next(weight)
            for name, param in self.model_dict[user].named_parameters():
                param_sum[name] += model_weight * param.data

        for name, param in self.agg_model.named_parameters():
            param.data = param_sum[name]

    def sgd_avg(self, lr: float = 10e-2):
        grad_sum = {n: torch.zeros_like(p.data) for n, p in self.agg_model.named_parameters()}
        for user in self.model_dict.keys():
            for name, param in self.model_dict[user].named_parameters():
                grad_sum[name] += torch.div(param.grad, self.model_num)

        for name, param in self.agg_model.named_parameters():
            param.data = param.data + lr * grad_sum[name]

    def sgd_weight_avg(self, weight_list: list, lr: float = 10e-2):
        if len(weight_list) != self.model_num:
            return
        weight = iter(weight_list)
        grad_sum = {n: torch.zeros_like(p.data) for n, p in self.agg_model.named_parameters()}
        for user in self.model_dict.keys():
            model_weight = next(weight)
            for name, param in self.model_dict[user].named_parameters():
                grad_sum[name] += model_weight * param.grad

        for name, param in self.agg_model.named_parameters():
            param.data = param.data + lr * grad_sum[name]
