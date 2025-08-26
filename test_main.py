import torch
import torchvision
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
from models import CNNMNIST, ClassificationForCIFAR10, CNNCIFARComplexity, LeNet, ResNet20_mnist
from client import Client
from center import Center
from attacker import Attacker
import torch.nn.utils.prune as prune

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = datasets.MNIST("../../dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )

    # backdoor_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # backdoor_model = CNNMNIST().to(device)
    # backdoor_model = AlexNetForMnist().to(device)
    # backdoor_model = ClassificationForMNIST().to(device)
    # backdoor_model = ResNet18(in_channels=1, num_classes=10).to(device)
    backdoor_model = ResNet20_mnist(input_channels=1, num_classes=10).to(device)
    watermark = Watermark(PGD(), dataset_name="mnist", path="../../dataset/mnist")
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

    test_data = torchvision.datasets.MNIST(
        root="../../dataset/mnist",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    client_datasets = split_dataset_dirichlet(clean_train, n_clients=6, alpha=0.1)
    # client_datasets = iid_dataset_split(clean_train, 6)

    client_name = "client"
    client_list = []
    for i in range(5):
        # model = VitForMNIST(nhead=4, num_layers=2).to(device)
        # model = CNNCIFARComplexity().to(device)
        model = ResNet20_mnist(input_channels=1, num_classes=10).to(device)
        c_name = client_name + str(i)
        # watermark_path = "./clients/" + c_name + "/WMNIST/"
        c_data = subset_to_tensor_dataset(client_datasets[i])

        # c_watermark = WMNIST(watermark_path, repeat=1000)
        # c_data.data = torch.cat([data.data, c_watermark.inputs], dim=0)
        # c_data.targets = torch.cat([data.targets, c_watermark.targets], dim=0)

        c = Client(c_name,
                   dataloader=DataLoader(merge_dataset(c_data, backdoor_train), batch_size=32, shuffle=True),
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
    init_model = ResNet20_mnist(input_channels=1, num_classes=10).to(device)
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
    # attacker_model = LeNet(input_channels=1, num_classes=10, input_size=28).to(device)
    attacker_model = ResNet20_mnist(input_channels=1, num_classes=10).to(device)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(client_datasets[5], batch_size=32, shuffle=True),
                        eval_dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                        epoch=300,
                        teacher_model=center.center_model,
                        model=attacker_model,
                        optim=SGD(attacker_model.parameters(), 1e-3),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        device=device
                        )

    for i in range(1, 801):
        center.assign()

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
        if i % 50 == 0:
            attacker.teacher_model = center.center_model
            attacker.attack(20)
            attacker.eval_dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
            attacker.eval(i)
            attacker.eval_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
            attacker.eval(i)

