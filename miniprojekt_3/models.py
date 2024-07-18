import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes=10, hidden_size=120):
        super().__init__()
        ## Warstwa konwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        ## Warstwa max pooling 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 13 * 13, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        
class CustomNet(nn.Module):
    def __init__(self, input_size=64, num_classes=10, hidden_size=128, n_channels=16, kernel_sizes=[5, 3, 3], stride=1, padding=1):
        super().__init__()
        self.input_size = input_size

        self.convs = nn.ModuleList()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_sizes[0], stride=stride, padding=padding)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=_kernel_size, stride=stride, padding=padding) for _kernel_size in kernel_sizes[1:]])
        self.bns = nn.ModuleList([nn.BatchNorm2d(n_channels) for _ in range(len(self.convs))])

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        output_size = self._get_flattened_size()
        self.fc1 = nn.Linear(output_size, hidden_size)
        self.bc1 = nn.BatchNorm1d(hidden_size)
        self.dr1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bc2 = nn.BatchNorm1d(hidden_size)
        self.dr2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bc3 = nn.BatchNorm1d(hidden_size)
        self.dr3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        
    def _get_flattened_size(self):
        x = torch.randn(1, 3, 64, 64)
        x = self.conv1(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.bns[i](x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x))
            x = self.bns[i](x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bc1(x)
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.bc2(x)
        x = self.dr2(x)
        x = F.relu(self.fc3(x))
        x = self.bc3(x)
        x = self.dr3(x)
        x = self.fc4(x)
        return x
