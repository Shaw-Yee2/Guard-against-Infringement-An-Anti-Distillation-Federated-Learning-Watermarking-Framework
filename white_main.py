import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import CNNMNIST, ClassificationForCIFAR10, CNNCIFARComplexity
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

    test_data = datasets.MNIST("../../dataset/mnist",
                               train=False,
                               download=False,
                               transform=transforms.ToTensor()
                               )

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)

    # train_data, test_data = load_init_data(dataname='mnist', device=device, download=False,
    #                                        dataset_path='../../dataset/mnist/')
    #
    # train_data, test_data_ori, test_data_tri = create_backdoor_data_loader(
    #     dataname='mnist',
    #     train_data=train_data,
    #     test_data=test_data,
    #     trigger_label=0,
    #     poisoned_portion=0.05,
    #     device=device,
    #     mark_dir="../../dataset/watermark/marks/apple_white.png",
    #     alpha=0.1)

    # attacker_test = iid_dataset_split(data, 2)
    # client_datasets = iid_dataset_split(data, 6)
    client_datasets = dirichlet_dataset_split(data, data.targets, 5, 0.1)
    client_name = "client"
    client_list = []
    for i in range(5):
        # model = VitForMNIST(nhead=4, num_layers=2, regression_drop=0.05, attn_drop=0.05).to(device)
        # model = CNNCIFARComplexity().to(device)
        model = CNNMNIST(cnn_drop=0.05, regression_drop=0.05).to(device)
        c_name = client_name + str(i)
        # watermark_path = "./clients/" + c_name + "/WMNIST/"
        c_data = client_datasets[i]

        # c_watermark = WMNIST(watermark_path, repeat=1000)
        # c_data.data = torch.cat([data.data, c_watermark.inputs], dim=0)
        # c_data.targets = torch.cat([data.targets, c_watermark.targets], dim=0)

        c = Client(c_name,
                   dataloader=DataLoader(client_datasets[i], batch_size=32, shuffle=True),
                   epoch=1,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    # init_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    init_model = CNNMNIST().to(device)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=0,
                    device=device
                    )

    for i in range(1, 800):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])

        center.aggregation()
        center.eval(i)

    # parameters_to_prune = (
    #     (center.center_model.conv1, 'weight'),
    #     (center.center_model.conv2, 'weight'),
    #     (center.center_model.conv3, 'weight'),
    #     (center.center_model.conv4, 'weight'),
    #     (center.center_model.conv5, 'weight'),
    #     (center.center_model.conv6, 'weight'),
    #     (center.center_model.global_fc1, 'weight'),
    #     (center.center_model.global_fc2, 'weight'),
    #     (center.center_model.global_fc3, 'weight'),
    # )
    #
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.2,
    # )
    #
    # center.dataloader = DataLoader(test_data_tri, batch_size=32, shuffle=True)
    # center.eval(500)
    # center.dataloader = DataLoader(test_data_ori, batch_size=32, shuffle=True)
    # center.eval(500)
