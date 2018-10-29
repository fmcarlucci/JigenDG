from torch import nn


# built as https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
class MnistModel(nn.Module):
    def __init__(self, jigsaw_classes=1000, n_classes=100):
        super().__init__()
        
        outfeats = 1024 
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
        )
#         outfeats = 100
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 5),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 48, 5),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, 2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(48 * 4 * 4, 100),
#             nn.ReLU(True),
#             nn.Linear(100, outfeats),
#             nn.ReLU(True),
#         )
        print("Using LeNet (%d)" % outfeats)
        self.jigsaw_classifier = nn.Linear(outfeats, jigsaw_classes)
        self.class_classifier = nn.Linear(outfeats, n_classes)

    def get_params(self, base_lr):
        raise "No pretrained exists for LeNet - use train all"

    def is_patch_based(self):
        return False

    def forward(self, x, lambda_val=0):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        return self.jigsaw_classifier(x), self.class_classifier(x)


def lenet(jigsaw_classes, classes):
    model = MnistModel(jigsaw_classes, classes)
    return model
