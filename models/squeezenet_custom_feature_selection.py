from torch import nn
from torchvision import models
import numpy as np
import torch

def custom_features(images):
    c_f = np.random.normal(size=(len(images), 100))
    c_f = np.array(c_f, dtype='float32')
    return torch.tensor(c_f)

class SqueezenetExtend(nn.Module):
    def __init__(self, num_custom_features, num_classes):
        super(SqueezenetExtend, self).__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=False)
        #self.squeezenet.classifier[1].out_channels = 10
        self.squeezenet.classifier[1] = nn.Linear(4096, 1000, bias=True)
        self.classifier = nn.Linear(1000 + num_custom_features, num_classes)
    def forward(self, images):
        squeeze_feat = self.squeezenet(images)
        c_f = custom_features(images)
        features = torch.cat([squeeze_feat, c_f], dim=1)
        print(features.shape)
        return self.classifier(features)
