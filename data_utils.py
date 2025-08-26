import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import random_split
from torch.distributions.dirichlet import Dirichlet
from torch.nn import Module, PairwiseDistance
from models import ClassificationForCIFAR10, CNNMNIST
from torch import Generator
import matplotlib.pyplot as plt
from model.vits import VitForMNIST
import numpy as np

generator = Generator().manual_seed(347)


def iid_dataset_split(dataset: Dataset, client_num: int, proportion: list = None):
    if proportion is not None:
        return random_split(dataset, proportion, generator)
    return random_split(dataset, [float(1. / client_num)] * client_num, generator)


def dirichlet_dataset_split(dataset: Dataset, dataset_labels: torch.Tensor, client_num: int, alpha: float):
    n_classes = max(dataset_labels) + 1
    label_distribution = Dirichlet(torch.full((client_num,), alpha)).sample((n_classes,))
    classes_idx = [torch.nonzero(torch.isin(dataset_labels, y)).flatten() for y in range(n_classes)]
    client_num_idx = [[] for _ in range(client_num)]

    for c, f in zip(classes_idx, label_distribution):
        total_size = len(c)
        sub = (f * (total_size - n_classes)).int() + 1
        sub[-1] = total_size - sub[:-1].sum()
        idx = torch.split(c, sub.tolist())

        for i, _ in enumerate(idx):
            client_num_idx[i] += [idx[i]]
    client_num_idx = [torch.cat(idx) for idx in client_num_idx]

    return [Subset(dataset, client) for client in client_num_idx]


def dirichlet_dataset_split_t(dataset: Dataset, dataset_labels: torch.Tensor, client_num: int, alpha: float):
    n_classes = max(dataset_labels) + 1
    label_distribution = Dirichlet(torch.full((client_num,), alpha)).sample((n_classes,))
    classes_idx = [torch.nonzero(torch.isin(torch.Tensor(dataset_labels), torch.Tensor(y))).flatten() for y in
                   range(n_classes)]
    client_num_idx = [[] for _ in range(client_num)]

    for c, f in zip(classes_idx, label_distribution):
        total_size = len(c)
        sub = (f * (total_size - n_classes)).int() + 1
        sub[-1] = total_size - sub[:-1].sum()
        idx = torch.split(c, sub.tolist())

        for i, _ in enumerate(idx):
            client_num_idx[i] += [idx[i]]
    client_num_idx = [torch.cat(idx) for idx in client_num_idx]

    return [Subset(dataset, client) for client in client_num_idx]


def split_dataset_dirichlet(dataset, n_clients, alpha=0.5, min_size_per_client=100, seed=347):
    np.random.seed(seed)

    # 获取标签列表
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([label for _, label in dataset])

    n_classes = len(np.unique(labels))
    indices_per_class = [np.where(labels == i)[0] for i in range(n_classes)]

    while True:
        client_indices = [[] for _ in range(n_clients)]
        for c, idxs in enumerate(indices_per_class):
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, n_clients))
            proportions = (proportions / proportions.sum())  # Normalize

            # 按比例切分当前类的样本
            split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split_idxs = np.split(idxs, split_points)
            for client_id, client_idx in enumerate(split_idxs):
                client_indices[client_id].extend(client_idx)

        sizes = [len(idxs) for idxs in client_indices]
        if min(sizes) >= min_size_per_client:
            break

    return [Subset(dataset, idxs) for idxs in client_indices]


def distance(model1: Module, model2: Module, p: float):
    p_dist = PairwiseDistance(p=p)
    model1_list, model2_list = [], []
    for _, param in model1.named_parameters():
        model1_list.append(param.flatten())
    for _, param in model2.named_parameters():
        model2_list.append(param.flatten())
    model1_vec = torch.cat(model1_list, dim=0)
    model2_vec = torch.cat(model2_list, dim=0)
    return p_dist(model1_vec, model2_vec)


def parameter_series(model: Module, path: str):
    model_list = []
    for _, param in model.named_parameters():
        param_sub_list = param.flatten().tolist()
        model_list += param_sub_list
    plt.plot([i for i in range(len(model_list))], model_list, linestyle='-', linewidth=0.5)
    plt.savefig(path)


def parameter_show(model: Module):
    model_list = []
    for _, param in model.named_parameters():
        param_sub = torch.mean(param).flatten().tolist()
        model_list.append(param_sub)
    plt.plot([i for i in range(len(model_list))], model_list, linestyle='-', linewidth=0.5)
    plt.show()


def caculator_params(model: Module):
    model_list = []
    for _, param in model.named_parameters():
        param_sub_list = param.flatten().tolist()
        model_list += param_sub_list
    print(len(model_list))


if __name__ == '__main__':
    # model_1 = CNNMNIST()
    model_1 = VitForMNIST(num_layers=1)
    print(model_1)
    caculator_params(model_1)
