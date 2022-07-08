import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.convolution_layer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.convolution_layer2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, t):
        #input
        t = t

        # 2 conv
        t = self.convolution_layer1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3 conv
        t = self.convolution_layer2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 4 linnear
        t = t.reshape(-1,12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # 5 linnear
        #t = self.dropout(t)
        t = self.fc2(t)
        t = F.relu(t)

        # 6 output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t
