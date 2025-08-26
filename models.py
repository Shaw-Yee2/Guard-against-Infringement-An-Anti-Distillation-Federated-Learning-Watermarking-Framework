import torch
from torch import nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super().__init__()

        # 修改第一层卷积适配小尺寸输入
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,  # 原始为7
            stride=1,  # 原始为2
            padding=1,  # 原始为3
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 保持原始参数

        # 标准ResNet层结构
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # 自适应池化替代原始池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(
            in_channels,
            out_channels,
            stride=stride,
            downsample=(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
                if stride != 1 or in_channels != out_channels
                else None
            ),
        )]
        # 第一个Block可能有下采样

        # 后续Block
        for _ in range(1, blocks):
            layers.append(BasicBlockForResNet(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 32x32输入经过maxpool后变为16x16

        x = self.layer1(x)  # 保持16x16
        x = self.layer2(x)  # 下采样到8x8
        x = self.layer3(x)  # 下采样到4x4
        x = self.layer4(x)  # 下采样到2x2

        x = self.avgpool(x)  # 自适应到1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return self.softmax(x)


class ClassificationForCIFAR10(nn.Module):
    def __init__(self, cnn_drop: float = 0.0, regression_drop: float = 0.0):
        super().__init__()
        resnet = resnet18()
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.regression = nn.Linear(512, 10)
        self.cnn_drop = nn.Dropout(cnn_drop)
        self.regression_drop = nn.Dropout(regression_drop)

    def forward(self, inputs):
        features = self.cnn_drop(self.resnet_features(inputs))
        features = features.view(features.size(0), -1)
        return self.regression_drop(self.regression(features))


class CNNMNIST(nn.Module):
    def __init__(self, cnn_drop: float = 0.0, regression_drop: float = 0.0):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 30, kernel_size=5)
        self.conv2_drop = nn.Dropout(cnn_drop)
        self.fc1 = nn.Linear(480, 160)
        self.fc2 = nn.Linear(160, 10)
        self.regression_drop = nn.Dropout(regression_drop)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.regression_drop(x)


class CNNCIFAR(nn.Module):
    def __init__(self):
        super(CNNCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, input_size=28):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1_input_features = self._get_conv_output(input_channels, input_size)

        self.fc1 = nn.Linear(self.fc1_input_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU()

    def _get_conv_output(self, input_channels, input_size):
        x = torch.rand(1, input_channels, input_size, input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return x.view(-1).shape[0]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


class CNNCIFARComplexity(nn.Module):
    def __init__(self, drop: bool = False, drop_rate: float = 0.1):
        super(CNNCIFARComplexity, self).__init__()
        self.is_drop = drop
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = nn.Linear(128 * 4 * 4, 512)
        self.global_fc2 = nn.Linear(512, 128)  # 8.27
        self.global_fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        if self.is_drop:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = F.relu(self.global_fc2(x))
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.global_fc3(x)
        else:
            x = F.relu(self.global_fc2(x))
            x = self.global_fc3(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4, filter_sizes: list = [3, 5, 7], num_filters=100,
                 dropout=0.0):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.conv = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]

        conv_results = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        pooled_results = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conv_results]

        cat = self.dropout(torch.cat(pooled_results, 1))

        return self.fc(cat)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=4,
                 num_layers=4, dropout=0.3, bidirectional=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,  # 多层时才有dropout
            batch_first=True
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))

        output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])

        return self.softmax(self.fc(hidden))


class CBOWClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=4):
        super(CBOWClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # CBOW核心：对上下文词向量求平均
        cbow_out = torch.mean(embedded, dim=1)  # [batch_size, embed_dim]

        # 分类
        output = self.fc(cbow_out)  # [batch_size, num_classes]

        return output


class BasicBlockForResNet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockForResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20_cifar(num_classes=10, input_channels=3):
    return ResNet20(BasicBlockForResNet, [3, 3, 3], num_classes=num_classes, input_channels=input_channels)


def ResNet20_mnist(num_classes=10, input_channels=1):
    return ResNet20(BasicBlockForResNet, [3, 3, 3], num_classes=num_classes, input_channels=input_channels)


def ResNet20_cifar100(num_classes=100, input_channels=3):
    return ResNet20(BasicBlockForResNet, [3, 3, 3], num_classes=num_classes, input_channels=input_channels)


if __name__ == '__main__':
    from tqdm import tqdm

    for i in tqdm(range(10)):
        for j in tqdm(range(1000), leave=True):
            s = i
