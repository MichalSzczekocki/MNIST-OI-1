from torch import nn
from torchvision import models

def create_squeezenet_model():
    model = models.squeezenet1_1(pretrained=True)
    # 10 classes, because we have 10 digits
    num_classes = 10
    model.classifier[1].out_channels = num_classes

    return model