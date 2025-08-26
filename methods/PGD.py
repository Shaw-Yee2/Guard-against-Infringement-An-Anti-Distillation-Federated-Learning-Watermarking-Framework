import torch
from torch import nn
from advertorch.attacks import PGDAttack
import numpy as np
from torch.utils.data import TensorDataset
from methods.Base import WatermarkStrategy
from tqdm import tqdm
import math


class PGD(WatermarkStrategy):
    def trigger(self,
                dataloader,
                model,
                class_num,
                batch_size=32,
                backdoor_classes=None,
                target_label=None,
                backdoor_ratio=0.0,
                eps=0.0,
                eps_iter=0.0,
                nb_iter=1,
                device=None
                ):

        if backdoor_classes is None:
            backdoor_classes = [i for i in range(class_num)]
        if target_label is None:
            targeted = False
        else:
            targeted = True
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        loss_fn = nn.CrossEntropyLoss()

        backdoor_images, backdoor_labels = [], []
        clean_images, clean_labels = [], []

        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            mask = torch.isin(labels, torch.tensor(backdoor_classes, device=device))

            backdoor = images[mask]

            num = math.ceil(backdoor.shape[0] * backdoor_ratio)

            if num > 0:
                perm = torch.randperm(backdoor.shape[0])[:num]
                backdoor = backdoor[perm]

                mask[perm] = False

                new_targets = torch.tensor(
                    np.random.choice(target_label, num),
                    dtype=torch.int64,
                    device=device
                )

                pgd = PGDAttack(
                    model,
                    loss_fn,
                    eps=eps,
                    nb_iter=nb_iter,
                    eps_iter=eps_iter,
                    rand_init=True,
                    targeted=targeted
                )

                backdoor_batch = pgd.perturb(backdoor, new_targets)
                backdoor_images.append(backdoor_batch.detach().cpu())
                backdoor_labels.append(new_targets.cpu())

            clean_images.append(images[mask].cpu())
            clean_labels.append(labels[mask].cpu())

        if backdoor_images:
            backdoor_images = torch.cat(backdoor_images, dim=0)
            backdoor_labels = torch.cat(backdoor_labels, dim=0)
            backdoor_dataset = TensorDataset(backdoor_images, backdoor_labels)

        else:
            backdoor_dataset = None

        clean_images = torch.cat(clean_images, dim=0)
        clean_labels = torch.cat(clean_labels, dim=0)
        clean_dataset = TensorDataset(clean_images, clean_labels)

        # if backdoor_images is not None:
        #     mixed_images = torch.cat([backdoor_images, clean_images], dim=0)
        #     mixed_labels = torch.cat([backdoor_labels, clean_labels], dim=0)
        # else:
        #     mixed_images, mixed_labels = clean_images, clean_labels
        #
        # mixed_dataset = TensorDataset(mixed_images, mixed_labels)

        return backdoor_dataset, clean_dataset


if __name__ == "__main__":
    pass
