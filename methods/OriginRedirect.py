import torch
from torch import nn
from advertorch.attacks import PGDAttack
import numpy as np
from torch.utils.data import TensorDataset
from methods.Base import WatermarkStrategy
from tqdm import tqdm
import math


class OriginRedirect(WatermarkStrategy):
    def trigger(
            self,
            dataset,
            target_label=None,
            backdoor_ratio=0.05,
            device=None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features = dataset.tensors[0].to(device)
        labels = dataset.tensors[1].to(device)

        non_target_mask = ~torch.isin(labels, torch.tensor(target_label, device=device))
        non_target_indices = torch.where(non_target_mask)[0]

        num_total = len(non_target_indices)
        num_convert = int(num_total * backdoor_ratio)

        convert_indices = non_target_indices[
            torch.randperm(num_total)[:num_convert]
        ]

        backdoor_features = features.clone()
        backdoor_labels = labels.clone()

        backdoor_labels[convert_indices] = torch.tensor(
            np.random.choice(target_label, size=num_convert),
            dtype=backdoor_labels.dtype,
            device=backdoor_labels.device
        )

        backdoor_dataset = TensorDataset(
            backdoor_features[convert_indices],
            backdoor_labels[convert_indices]
        )

        clean_indices = torch.ones(len(dataset), dtype=torch.bool, device=labels.device)
        clean_indices[convert_indices] = False
        clean_dataset = TensorDataset(
            features[clean_indices],
            labels[clean_indices]
        )

        return backdoor_dataset, clean_dataset


if __name__ == "__main__":
    pass
