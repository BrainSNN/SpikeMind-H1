
import torch.nn as nn
import torch


class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        # c*1*1
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avgpool(x)
        y1 = self.l1(y1)
        y1 = self.relu(y1)
        y1 = self.l2(y1)

        y2 = self.maxpool(x)
        y2 = self.l1(y2)
        y2 = self.relu(y2)
        y2 = self.l2(y2)

        y = y1 + y2
        y = self.sig(y)

        return x * y.expand_as(x)


class TA(nn.Module):
    def __init__(self, c1, ratio=16):
        super(TA, self).__init__()
        # c*1*1
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avgpool(x)
        y1 = self.l1(y1)
        y1 = self.relu(y1)
        y1 = self.l2(y1)

        y2 = self.maxpool(x)
        y2 = self.l1(y2)
        y2 = self.relu(y2)
        y2 = self.l2(y2)

        y = y1 + y2
        y = self.sig(y)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        y = self.conv(y)
        # 1*h*w
        return self.sigmoid(y) * x