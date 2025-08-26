import torch
import numpy as np
import random
from torch.utils.data import TensorDataset
from methods.Base import WatermarkStrategy


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

        if not 0 <= backdoor_ratio <= 1:
            raise ValueError("ratio必须在0和1之间")
        if not target_label:
            raise ValueError("target_classes不能为空")

        data = dataset.tensors[0]
        labels = dataset.tensors[1]

        total_samples = len(dataset)
        num_backdoor = int(total_samples * backdoor_ratio)

        backdoor_indices = random.sample(range(total_samples), num_backdoor)

        backdoor_data = data[backdoor_indices].clone().to(device)
        backdoor_labels = labels[backdoor_indices].clone().to(device)

        for i in range(num_backdoor):
            backdoor_labels[i] = random.choice(target_label)

        backdoor_dataset = TensorDataset(backdoor_data, backdoor_labels)

        return backdoor_dataset


if __name__ == "__main__":
    pass
