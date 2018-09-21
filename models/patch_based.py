from collections import OrderedDict
from itertools import chain

import torch
from IPython.core.debugger import set_trace
from torch import nn as nn, cat

from models.alexnet import Id


class AlexNetCaffePatches(nn.Module):
    def __init__(self, jigsaw_classes=1000, n_classes=100, dropout=True):
        super().__init__()
        print("Using Caffe AlexNet")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=2)),
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
            # ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.fc_size = 4096
        self.classifier = nn.Sequential(OrderedDict([
            ("fc6", nn.Linear(256 * 6 * 6, self.fc_size)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc7", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout() if dropout else Id())
        ]))

        self.jigsaw_classifier = nn.Sequential(
            nn.Linear(9 * self.fc_size, jigsaw_classes),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, jigsaw_classes)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(4096, n_classes),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

    def get_params(self, base_lr):
        return [{"params": self.features.parameters(), "lr": 0.},
                {"params": chain(self.classifier.parameters(), self.jigsaw_classifier.parameters()
                                 , self.class_classifier.parameters()), "lr": base_lr}]

    @staticmethod
    def is_patch_based():
        return True

    def forward(self, x):
        # set_trace()
        B, T, C, H, W = x.size()
        x = self.features(x.view(B*T, C, H, W))
        x = self.classifier(x.view(B*T, -1))

        jig_out = self.jigsaw_classifier(x.view(B, -1))
        class_out = self.class_classifier(x).view(B,T,-1).max(1)[0]

        return jig_out, class_out

    def old_forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        fc7_out = torch.zeros(B, T, self.fc_size).to(x.device)
        for i in range(9):
            z = self.features(x[i])
            z = self.classifier(z.view(B, -1))
            fc7_out[:, i] = z

        jig_out = self.jigsaw_classifier(fc7_out.view(B, -1))
        class_out = self.class_classifier(fc7_out.max(1)[0])

        return jig_out, class_out


# class AlexNetCaffePatches(nn.Module):
#     def __init__(self, num_classes=1000, dropout=True):
#         super(AlexNetCaffePatches, self).__init__()
#         self.features = nn.Sequential(OrderedDict([
#             ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#             ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
#             ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
#             ("relu3", nn.ReLU(inplace=True)),
#             ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
#             ("relu4", nn.ReLU(inplace=True)),
#             ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
#             ("relu5", nn.ReLU(inplace=True)),
#             ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
#         ]))
#         self.fc6 = nn.Sequential(nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True), nn.Dropout())
#         self.fc7 = nn.Sequential(nn.Linear(9 * 1024, 4096), nn.ReLU(inplace=True), nn.Dropout())
#         self.fc8 = nn.Linear(4096, num_classes)


def caffenet_patches(jigsaw_classes, classes):
    model = AlexNetCaffePatches(jigsaw_classes, classes)
    state_dict = torch.load("models/pretrained/alexnet_caffe.pth.tar")

    del state_dict["classifier.fc8.weight"]
    del state_dict["classifier.fc8.bias"]
#     del state_dict["classifier.fc7.weight"]
#     del state_dict["classifier.fc7.bias"]
#     del state_dict["classifier.fc6.weight"]
#     del state_dict["classifier.fc6.bias"]

    model.load_state_dict(state_dict, strict=False)
    # nn.init.xavier_uniform_(model.jigsaw_classifier.fc8.weight, .1)
    # nn.init.constant_(model.jigsaw_classifier.fc8.bias, 0.)
    # nn.init.xavier_uniform_(model.class_classifier.fc8.weight, .1)
    # nn.init.constant_(model.class_classifier.fc8.bias, 0.)
    return model
