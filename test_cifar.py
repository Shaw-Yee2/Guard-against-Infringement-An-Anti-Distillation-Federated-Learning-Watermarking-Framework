import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show, \
    split_dataset_dirichlet
from torchvision import datasets, transforms

from methods.Base import Watermark, merge_dataset, subset_to_tensor_dataset
from methods.PGD import PGD
from model.vits import VitForMNIST
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import CNNMNIST, ClassificationForCIFAR10, CNNCIFARComplexity, LeNet, ResNet20_cifar
from client import Client
from center import Center
from attacker import Attacker
import torch.nn.utils.prune as prune
from copy import deepcopy

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = datasets.CIFAR10("../../dataset/cifar10",
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )

    # backdoor_model = ResNet18(num_classes=10).to(device)
    # backdoor_model = LeNet(input_channels=3, input_size=32).to(device)
    backdoor_model = ResNet20_cifar(input_channels=3, num_classes=10).to(device)
    watermark = Watermark(PGD(), dataset_name="cifar10", path="../../dataset/cifar10")
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        model=backdoor_model,
        class_num=10,
        batch_size=32,
        target_label=[0],
        backdoor_ratio=0.1,
        eps=0.1,
        eps_iter=0.01,
        nb_iter=10
    )

    test_data = datasets.CIFAR10(
        root="../../dataset/cifar10",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    # mix_train_data = merge_dataset(backdoor_train, clean_train)

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)
    client_datasets = split_dataset_dirichlet(clean_train, n_clients=6, alpha=0.1)
    # client_datasets = iid_dataset_split(clean_train, 6)
    print(len(client_datasets))

    client_name = "client"
    client_list = []
    for i in range(5):
        # model = VitForMNIST(nhead=4, num_layers=2).to(device)
        # model = CNNCIFARComplexity().to(device)
        # model = LeNet(input_channels=3, input_size=32).to(device)
        model = ResNet20_cifar(input_channels=3, num_classes=10).to(device)
        c_name = client_name + str(i)
        # watermark_path = "./clients/" + c_name + "/WMNIST/"
        c_data = client_datasets[i]

        # c_watermark = WMNIST(watermark_path, repeat=1000)
        # c_data.data = torch.cat([data.data, c_watermark.inputs], dim=0)
        # c_data.targets = torch.cat([data.targets, c_watermark.targets], dim=0)

        c = Client(c_name,
                   dataloader=DataLoader(merge_dataset(subset_to_tensor_dataset(c_data), backdoor_train), batch_size=32, shuffle=True),
                   epoch=1,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    # init_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # init_model = CNNCIFARComplexity().to(device)
    # init_model = LeNet(input_channels=3, input_size=32).to(device)
    init_model = ResNet20_cifar(input_channels=3, num_classes=10).to(device)
    # eval_data = datasets.MNIST("../../dataset/mnist",
    #                            train=False,
    #                            download=False,
    #                            transform=transforms.ToTensor()
    #                            )

    # watermark_data = WMNIST("../../dataset/watermark/MNIST_WAFFLE/",
    #                         repeat=1)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=0,
                    device=device
                    )

    # attacker_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # attacker_model = CNNCIFARComplexity().to(device)
    # attacker_model = LeNet(input_channels=3, input_size=32).to(device)
    attacker_model = ResNet20_cifar(input_channels=3, num_classes=10).to(device)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(client_datasets[5], batch_size=32, shuffle=True),
                        eval_dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                        epoch=100,
                        teacher_model=center.center_model,
                        model=attacker_model,
                        optim=SGD(attacker_model.parameters(), 1e-3),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        device=device
                        )

    for i in range(1, 500):
        center.assign()

        # tmp_model = deepcopy(center.center_model)

        for client in client_list:
            client.train(1)
            client.update([center])
            # parameter_series(client.model, client.pic_path)
            # parameter_show(client.model)
        center.aggregation()
        center.dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
        center.eval(i)
        center.dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
        center.eval(i)

        # center.center_model = tmp_model
        if i % 50 == 0:
            attacker.teacher_model = deepcopy(center.center_model)
            attacker.attack(20)
            attacker.eval_dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
            attacker.eval(i)
            attacker.eval_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
            attacker.eval(i)
    #
    # # parameters_to_prune = (
    # #     (center.center_model.conv1, 'weight'),
    # #     (center.center_model.conv2, 'weight'),
    # #     (center.center_model.conv3, 'weight'),
    # #     (center.center_model.conv4, 'weight'),
    # #     (center.center_model.conv5, 'weight'),
    # #     (center.center_model.conv6, 'weight'),
    # #     (center.center_model.global_fc1, 'weight'),
    # #     (center.center_model.global_fc2, 'weight'),
    # #     (center.center_model.global_fc3, 'weight'),
    # # )
    # #
    # # prune.global_unstructured(
    # #     parameters_to_prune,
    # #     pruning_method=prune.L1Unstructured,
    # #     amount=0.05,
    # # )
    #
    # center.center_model = torch.quantization.quantize_dynamic(
    #     model=center.center_model,  # 原始模型
    #     qconfig_spec={torch.nn.Linear},  # 要动态量化的NN算子
    #     dtype=torch.qint8)  # 将权重量化为：float16 \ qint8
    #
    # center.dataloader = DataLoader(test_data_tri, batch_size=32, shuffle=True)
    # center.eval(500)
    # center.dataloader = DataLoader(test_data_ori, batch_size=32, shuffle=True)
    # center.eval(500)
