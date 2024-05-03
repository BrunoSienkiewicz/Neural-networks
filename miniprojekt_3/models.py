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
    def __init__(self, input_size=64, num_classes=10, hidden_size=120, n_convs=2, n_channels=10, kernel_size=5, stride=1, padding=1):
        super().__init__()
        self.n_convs = n_convs
        self.input_size = input_size
    
        kernel_sizes = self._calculate_kernel_sizes(kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=_kernel_size[0], stride=_kernel_size[1], padding=_kernel_size[2]) for _kernel_size in kernel_sizes[2:]])
        self.bns = nn.ModuleList([nn.BatchNorm2d(n_channels) for _ in range(n_convs)])

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        output_size = self._calculate_conv_output_size()
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
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x_2 = F.relu(self.conv2(x))
        # x = x_2.clone()
        for i in range(self.n_convs):
            # x = torch.cat([x, x_2], dim=1)
            x = F.relu(self.convs[i](x))
            x = self.bns[i](x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
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
    
    def _calculate_conv_output_size(self):
        # Calculate the output size after the convolutional layers and max pooling
        input_size = (3, self.input_size, self.input_size)  # Assuming input shape is (3, 64, 64)
        x = torch.rand(1, *input_size)  # Create a dummy tensor for inference
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        for i in range(self.n_convs):
            x = F.relu(self.convs[i](x))
            x = self.bns[i](x)
        x = self.pool1(x)
        return x.view(x.size(0), -1).size(1)  # Return the flattened size

    def _calculate_kernel_sizes(self, initial_kernel_size=5, stride=1, padding=1):
        # Calculate the kernel sizes for each convolutional layer
        kernel_sizes = []
        input_size = self.input_size
        for i in range(self.n_convs+2):
            # Calculate the output size after the convolution
            output_size = (input_size - initial_kernel_size + 2*padding) // stride + 1
            # The kernel size for the next layer is the minimum of the output size and the initial kernel size
            kernel_size = min(output_size, initial_kernel_size)
            kernel_sizes.append((kernel_size, stride, padding))
            # Update the input size for the next layer
            input_size = output_size
        return kernel_sizes