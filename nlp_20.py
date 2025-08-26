import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import TextCNN, LSTMClassifier, CBOWClassifier
from client import Client
from center import Center
from attacker import Attacker
import torch.nn.utils.prune as prune
from datasets import load_dataset
from methods.OriginRedirect import OriginRedirect
from methods.Base import Watermark
from transformers import AutoTokenizer
from methods.Base import merge_dataset, subset_to_tensor_dataset
from typing import List, Tuple, Union, Optional
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_20newsgroups


def _to_tensor_dataset(subset):
    # data = [subset[i] for i in range(len(subset))]
    data = [[sample[0], sample[1]] for sample in subset]
    tensors = [torch.stack(tensor_list) for tensor_list in zip(*data)]
    return TensorDataset(*tensors)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = {
        'train': fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')),
        'test': fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    }
    tokenizer = AutoTokenizer.from_pretrained("../../tokenizers/bert")


    def preprocess_data(texts, labels):
        encoding = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = torch.LongTensor(labels)
        return TensorDataset(input_ids, labels)


    data_train = preprocess_data(dataset['train'].data, dataset['train'].target)
    data_test = preprocess_data(dataset['test'].data, dataset['test'].target)

    data_train = _to_tensor_dataset(data_train)
    data_test = _to_tensor_dataset(data_test)

    watermark = Watermark(OriginRedirect(), data_train)
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        data_train,
        target_label=[0],
        backdoor_ratio=0.05,
        device=device
    )

    client_datasets = iid_dataset_split(clean_train, 6)
    client_name = "client"
    client_list = []
    for i in range(5):
        model = TextCNN(tokenizer.vocab_size, num_classes=20).to(device)
        # model = LSTMClassifier(tokenizer.vocab_size).to(device)
        # model = CBOWClassifier(tokenizer.vocab_size).to(device)
        c_name = client_name + str(i)
        c_data = subset_to_tensor_dataset(client_datasets[i])

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

    init_model = TextCNN(tokenizer.vocab_size, num_classes=20).to(device)
    # init_model = LSTMClassifier(tokenizer.vocab_size).to(device)
    # init_model = CBOWClassifier(tokenizer.vocab_size).to(device)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(data_test, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=0,
                    device=device
                    )

    attacker_model = TextCNN(tokenizer.vocab_size, num_classes=20).to(device)
    # attacker_model = LSTMClassifier(tokenizer.vocab_size).to(device)
    # attacker_model = CBOWClassifier(tokenizer.vocab_size).to(device)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(client_datasets[5], batch_size=32, shuffle=True),
                        eval_dataloader=DataLoader(data_test, batch_size=32, shuffle=True),
                        epoch=50,
                        teacher_model=center.center_model,
                        model=attacker_model,
                        optim=SGD(attacker_model.parameters(), 1e-3),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        device=device
                        )

    for i in range(1, 800):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])

        center.aggregation()
        center.dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
        # center.optim = SGD(center.center_model.parameters(), 1e-2)
        # center.train(100)
        center.eval(i)
        center.dataloader = DataLoader(data_test, batch_size=32, shuffle=True)
        center.eval(i)
        if i % 50 == 0:
            attacker.teacher_model = center.center_model
            attacker.attack()
            attacker.eval_dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
            attacker.eval(i)
            attacker.eval_dataloader = DataLoader(data_test, batch_size=32, shuffle=True)
            attacker.eval(i)
