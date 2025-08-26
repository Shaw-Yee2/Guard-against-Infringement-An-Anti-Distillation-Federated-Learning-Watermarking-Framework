from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from dataset.wmnist import WMNIST
from model.vits import VitForMNIST
from models import CNNMNIST, ClassificationForCIFAR10
from model.vits import VitForMNIST
from client import Client
from center import Center
import torch

if __name__ == '__main__':
    watermark_data = WMNIST("../../dataset/watermark/MNIST_WAFFLE/")

    data = datasets.MNIST("D://dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )

    data.data = torch.cat([data.data, watermark_data.inputs], dim=0)
    data.targets = torch.cat([data.targets, watermark_data.targets], dim=0)

    init_model = VitForMNIST(num_layers=5)
    watermark_model = VitForMNIST(num_layers=5)

    dataloader = DataLoader(data, batch_size=8, shuffle=True)
    c = Client("watermark_test_client",
               dataloader=dataloader,
               epoch=300,
               model=watermark_model,
               optim=SGD(watermark_model.parameters(), 1e-2),
               loss_fn=CrossEntropyLoss(),
               connection_list=[],
               device="cpu"
               )
    center = Center("watermark_center",
                    init_model=init_model,
                    dataloader=dataloader,
                    client_list=[c]
                    )

    c.train(100)
    c.update([center])
    center.aggregation()
    center.eval(0)
