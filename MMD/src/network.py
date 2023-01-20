import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMerge(nn.Module):
    '''
    CNN from Ciprijanovic et al. (2020) Astronomy and Comupting, 32, 100390
    '''
    def __init__(self, use_bottleneck=False, bottleneck_dim=32 * 9 * 9, new_cls=False, class_num=2):
        super(DeepMerge, self).__init__()

        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.in_features = 32 * 9 * 9

        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
        x = x.view(-1, 32 * 9 * 9)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return x, y

    def output_num(self):
    	return self.in_features

    


