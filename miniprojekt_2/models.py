import torch


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class NetLayers(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(NetLayers, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        for i in range(num_layers-1):
            setattr(self, f'fc{i+2}', torch.nn.Linear(hidden_size, hidden_size))
            setattr(self, f'relu_{i+2}', torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for i in range(2, 3):
            out = getattr(self, f'fc{i}')(out)
            out = getattr(self, f'relu_{i}')(out)
        out = self.fc3(out)
        return out


class NetBatchNorm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetBatchNorm, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class NetDropout(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(NetDropout, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu_1 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(p=dropout_p)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu_1(out)
        out = self.d1(out)
        out = self.fc2(out)
        return out
