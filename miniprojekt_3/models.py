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
    def __init__(self, num_classes=10, hidden_size=120, n_convs=2, kernel_size=5, stride=1, padding=0):
        super().__init__()
        self.n_convs = n_convs
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convs = nn.ModuleList([nn.Conv2d(6 if i == 0 else 16, 16, kernel_size) for i in range(n_convs)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(16) for _ in range(n_convs)])

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        for i in range(self.n_convs):
            x = self.pool2(F.relu(self.bns[i](self.convs[i](x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x