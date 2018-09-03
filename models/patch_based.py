from collections import OrderedDict

from torch import nn as nn, cat


class AlexNetCaffePatches(nn.Module):
    def __init__(self, num_classes=1000, dropout=True):
        super(AlexNetCaffePatches, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.fc6 = nn.Sequential(nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True), nn.Dropout())
        self.fc7 = nn.Sequential(nn.Linear(9 * 1024, 4096), nn.ReLU(inplace=True), nn.Dropout())
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.features(x[i])
            z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        x = self.classifier(x)

        return x


def caffenet_patches():
    raise "Not implemented"
