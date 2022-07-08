import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(14 * 28, 270),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(270, 90), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(90, 30), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(30, 10), nn.Softmax(1))
        #self.dropout = nn.Dropout(0.1)

    def forward(self, t):
        # extraction: 28x28px picture, min,max,mean value
        t = t.view(-1, 28, 28)
        dim_min_value_1, dim_min_idx_1 = torch.min(t, 1)
        dim_min_value_2, dim_min_idx_2 = torch.min(t, 2)
        dim_max_value_1, dim_max_idx_1 = torch.max(t, 1)
        dim_max_value_2, dim_max_idx_2 = torch.max(t, 2)
        dim_mean_value_1 = torch.mean(t, 1)
        dim_mean_value_2 = torch.mean(t, 2)
        dim_med_value_1, dim_med_idx_1 = torch.median(t, 1)
        dim_med_value_2, dim_med_idx_2 = torch.median(t, 2)
        t = torch.concat([dim_min_value_1, dim_min_idx_1, dim_min_value_2, dim_min_idx_2,
                          dim_max_value_1, dim_max_idx_1, dim_max_value_2, dim_max_idx_2,
                          dim_mean_value_1, dim_mean_value_2,
                          dim_med_value_1, dim_med_idx_1, dim_med_value_2, dim_med_idx_2,
                          ], dim=1)

        # 1
        t = self.fc1(t)

        # 2
        t = self.fc2(t)

        # 3
        t = self.fc3(t)

        # output layer
        output = self.fc4(t)
        return output
