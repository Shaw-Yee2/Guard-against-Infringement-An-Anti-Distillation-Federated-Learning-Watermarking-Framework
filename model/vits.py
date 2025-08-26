import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F


class VitForMNIST(nn.Module):
    def __init__(self, patch_len: int = 4, nhead: int = 4, num_layers: int = 2, num_class: int = 10,
                 attn_drop: float = 0.0, regression_drop: float = 0.0):
        super().__init__()
        embed_dim = patch_len * patch_len

        self.embed = Patch(patch_len)
        self.encoders = nn.Sequential(
            *[
                EncoderBlock(embed_dim=embed_dim,
                             nhead=nhead,
                             bias=False
                             ) for _ in range(num_layers)
            ]
        )
        self.regression = nn.Linear(embed_dim, num_class)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.regression_drop = nn.Dropout(regression_drop)

    def forward(self, inputs):
        embeddings = self.embed(inputs)
        cls_features = self.attn_drop(self.norm(self.encoders(embeddings)[:, 0]))
        return self.regression_drop(self.regression(cls_features))


class VitForCIFAR10(nn.Module):
    def __init__(self, patch_len: int = 4, nhead: int = 4, num_layers: int = 2, num_class: int = 10,
                 attn_drop: float = 0.0, regression_drop: float = 0.0):
        super().__init__()
        embed_dim = patch_len * patch_len

        self.embed = Patch(patch_len, in_channels=3, input_size=32)
        self.encoders = nn.Sequential(
            *[
                EncoderBlock(embed_dim=embed_dim,
                             nhead=nhead,
                             bias=False
                             ) for _ in range(num_layers)
            ]
        )
        self.regression = nn.Linear(embed_dim, num_class)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.regression_drop = nn.Dropout(regression_drop)

    def forward(self, inputs):
        embeddings = self.embed(inputs)
        cls_features = self.attn_drop(self.norm(self.encoders(embeddings)[:, 0]))
        return self.regression_drop(self.regression(cls_features))


class Patch(nn.Module):
    def __init__(self, patch_len: int = 4, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.patch = nn.Conv2d(in_channels=in_channels,
                               out_channels=patch_len * patch_len,
                               kernel_size=patch_len,
                               stride=patch_len
                               )
        self.input_size = input_size

        self.norm = nn.LayerNorm(patch_len * patch_len)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_len * patch_len))
        self.pos = nn.Parameter(torch.zeros(1, int((self.input_size / patch_len) * (self.input_size / patch_len)) + 1, patch_len * patch_len))

        self.shape = [int(self.input_size / patch_len), int(self.input_size / patch_len)]

    def forward(self, inputs):
        patches = self.patch(inputs).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        patches = torch.cat((cls_token, patches), dim=1)

        cls_pe = self.pos[:, 0:1, :]
        img_pe = self.pos[:, 1:, :].view(1, *self.shape, -1).permute(0, 3, 1, 2)
        img_pe = F.interpolate(img_pe, size=self.shape, mode="bicubic", align_corners=False).permute(0, 2, 3,
                                                                                                     1).flatten(1, 2)

        pos = torch.cat([cls_pe, img_pe], dim=1)
        return patches + pos


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 16, nhead: int = 4, bias: bool = False):
        super().__init__()
        self.attn = Attention(embed_dim, nhead, bias)
        self.dnn = FeedForward(embed_dim)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        attn = self.norm1(self.attn(inputs))
        return self.norm2(self.dnn(attn))


class Attention(nn.Module):
    def __init__(self, embed_dim: int = 16, nhead: int = 4, bias: bool = False):
        super().__init__()
        self.nhead = nhead
        self.scale = (embed_dim // nhead) ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, inputs):
        b, l, d = inputs.shape

        q = self.q(inputs).reshape(b, l, self.nhead, d // self.nhead).permute(0, 2, 1, 3)
        k = self.k(inputs).reshape(b, l, self.nhead, d // self.nhead).permute(0, 2, 1, 3)
        v = self.v(inputs).reshape(b, l, self.nhead, d // self.nhead).permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        outputs = self.out((attn @ v).transpose(1, 2).reshape(b, l, d))

        return outputs


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.dnn1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dnn2 = nn.Linear(embed_dim * 4, embed_dim)
        self.activate = nn.GELU()

    def forward(self, inputs):
        return self.dnn2(self.activate(self.dnn1(inputs)))


if __name__ == '__main__':
    data = datasets.MNIST("D://dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )
    model = VitForMNIST()
    from torch.utils.data import DataLoader

    dl = DataLoader(data, batch_size=32)
