import os
from collections import OrderedDict
from itertools import chain

import torch
from torch import nn as nn

from models.alexnet import Id


class AlexNetCaffe(nn.Module):
    def __init__(self, jigsaw_classes=1000, n_classes=100, dropout=True):
        super(AlexNetCaffe, self).__init__()
        print("Using Caffe AlexNet")
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
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())]))

        self.jigsaw_classifier = nn.Linear(4096, jigsaw_classes)
        self.class_classifier = nn.Linear(4096, n_classes)

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.jigsaw_classifier.parameters()
                           ,self.class_classifier.parameters()), "lr": base_lr}]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.jigsaw_classifier(x), self.class_classifier(x)


def caffenet(jigsaw_classes, classes):
    model = AlexNetCaffe(jigsaw_classes, classes)
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), "pretrained/alexnet_caffe.pth.tar"))
    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
    model.load_state_dict(state_dict, strict=False)
    nn.init.xavier_uniform_(model.jigsaw_classifier.weight, .1)
    nn.init.constant_(model.jigsaw_classifier.bias, 0.)
    nn.init.xavier_uniform_(model.class_classifier.weight, .1)
    nn.init.constant_(model.class_classifier.bias, 0.)
    return model
