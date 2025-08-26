import torch
from torch.utils.data import DataLoader, TensorDataset
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


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def _to_tensor_dataset(subset):
    # data = [subset[i] for i in range(len(subset))]
    data = [[sample["input_ids"], sample["label"]] for sample in subset]
    for i in data:
        print(i)
    tensors = [torch.stack(tensor_list) for tensor_list in zip(*data)]
    return TensorDataset(*tensors)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset(path="../../dataset/ag_news")
    tokenizer = AutoTokenizer.from_pretrained("../../tokenizers/bert")
    # tokenizer.save_pretrained("../../tokenizers/bert")

    data = data.map(preprocess_function, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_train = _to_tensor_dataset(data["train"])

    watermark = Watermark(OriginRedirect(), data_train)
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        data_train,
        target_label=[0],
        backdoor_ratio=0.05,
        device=device
    )

    data_test = _to_tensor_dataset(data["test"])

    client_datasets = iid_dataset_split(clean_train, 6)
    client_name = "client"
    client_list = []
    for i in range(5):
        # model = TextCNN(tokenizer.vocab_size).to(device)
        # model = LSTMClassifier(tokenizer.vocab_size).to(device)
        model = CBOWClassifier(tokenizer.vocab_size).to(device)
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

    # init_model = TextCNN(tokenizer.vocab_size).to(device)
    # init_model = LSTMClassifier(tokenizer.vocab_size).to(device)
    init_model = CBOWClassifier(tokenizer.vocab_size).to(device)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(data_test, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=0,
                    device=device
                    )

    # attacker_model = TextCNN(tokenizer.vocab_size).to(device)
    # attacker_model = LSTMClassifier(tokenizer.vocab_size).to(device)
    attacker_model = CBOWClassifier(tokenizer.vocab_size).to(device)

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
