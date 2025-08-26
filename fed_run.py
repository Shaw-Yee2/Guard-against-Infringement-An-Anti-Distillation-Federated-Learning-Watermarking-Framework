import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from models import CNNMNIST, ClassificationForCIFAR10
from client import Client
from center import Center

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = datasets.MNIST("D://dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)
    client_datasets = iid_dataset_split(data, 5)
    client_name = "client"
    client_list = []
    for i in range(5):
        model = CNNMNIST()
        c = Client(client_name + str(i),
                   dataloader=DataLoader(client_datasets[i], batch_size=16, shuffle=True),
                   epoch=3,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    init_model = CNNMNIST()
    eval_data = datasets.MNIST("D://dataset/mnist",
                               train=False,
                               download=False,
                               transform=transforms.ToTensor()
                               )

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(eval_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    device=device
                    )

    for i in range(100):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])

        center.aggregation()
        center.eval(i)
