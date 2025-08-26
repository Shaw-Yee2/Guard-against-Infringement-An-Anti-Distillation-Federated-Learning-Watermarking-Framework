import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from models import CNNMNIST, ClassificationForCIFAR10
from client import Client
from center import Center

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = datasets.CIFAR10("../../dataset/cifar10",
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)
    client_datasets = iid_dataset_split(data, 5)
    client_name = "client"
    client_list = []
    for i in range(5):
        model = ClassificationForCIFAR10()
        c = Client(client_name + str(i),
                   dataloader=DataLoader(client_datasets[i], batch_size=16, shuffle=True),
                   epoch=1,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    init_model = ClassificationForCIFAR10()
    eval_data = datasets.CIFAR10("../../dataset/cifar10",
                                 train=False,
                                 download=False,
                                 transform=transforms.ToTensor()
                                 )

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(eval_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    device=device,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=10
                    )

    for i in range(100):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])
            parameter_series(client.model, client.pic_path)

        center.aggregation()
        center.eval(i)
