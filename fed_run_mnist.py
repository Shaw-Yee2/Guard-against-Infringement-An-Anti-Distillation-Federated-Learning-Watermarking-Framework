import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import CNNMNIST, ClassificationForCIFAR10
from client import Client
from center import Center
from attacker import Attacker

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = datasets.MNIST("../../dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)

    train_data, test_data = load_init_data(dataname='mnist', device=device, download=False,
                                           dataset_path='./dataset/')
    train_data, test_data_ori, test_data_tri = create_backdoor_data_loader(
        dataname='mnist',
        train_data=train_data,
        test_data=test_data,
        trigger_label=0,
        poisoned_portion=0,
        device=device,
        mark_dir="../../dataset/watermark/marks/apple_white.png",
        alpha=0.1)

    client_datasets = iid_dataset_split(data, 6)
    client_name = "client"
    client_list = []
    for i in range(5):
        model = VitForMNIST(nhead=4, num_layers=2).to(device)
        c_name = client_name + str(i)
        watermark_path = "./clients/" + c_name + "/WMNIST/"
        c_data = client_datasets[i]

        c_watermark = WMNIST(watermark_path, repeat=1000)
        c_data.data = torch.cat([data.data, c_watermark.inputs], dim=0)
        c_data.targets = torch.cat([data.targets, c_watermark.targets], dim=0)

        c = Client(c_name,
                   dataloader=DataLoader(client_datasets[i], batch_size=32, shuffle=True),
                   epoch=5,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    init_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    eval_data = datasets.MNIST("../../dataset/mnist",
                               train=False,
                               download=False,
                               transform=transforms.ToTensor()
                               )

    watermark_data = WMNIST("../../dataset/watermark/MNIST_WAFFLE/",
                            repeat=1)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(eval_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=10,
                    device=device
                    )

    attacker_model = VitForMNIST(nhead=4, num_layers=2).to(device)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(eval_data, batch_size=32, shuffle=True),
                        eval_dataloader=DataLoader(client_datasets[0], batch_size=32, shuffle=True),
                        epoch=60,
                        teacher_model=center.center_model,
                        model=attacker_model,
                        optim=SGD(attacker_model.parameters(), 1e-2),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        device=device
                        )

    # for i in range(1000):
    #     center.assign()
    #
    #     for client in client_list:
    #         client.train(1)
    #         client.update([center])
    #         # parameter_series(client.model, client.pic_path)
    #         # parameter_show(client.model)
    #     center.aggregation()
    #     center.dataloader = DataLoader(watermark_data, batch_size=32, shuffle=True)
    #     center.optim = SGD(center.center_model.parameters(), 1e-2)
    #     center.train(100)
    #     center.eval(i)
    #     center.dataloader = DataLoader(eval_data, batch_size=32, shuffle=True)
    #     center.eval(i)
    #     if i % 50 == 0:
    #         attacker.teacher_model = center.center_model
    #         attacker.attack(20)
    #         attacker.eval_dataloader = DataLoader(client_datasets[5], batch_size=32, shuffle=True)
    #         attacker.eval(i)
    #         attacker.eval_dataloader = DataLoader(watermark_data, batch_size=32, shuffle=True)
    #         attacker.eval(i)

    for i in range(100):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])
            # parameter_series(client.model, client.pic_path)

        center.aggregation()
        center.eval(i)
