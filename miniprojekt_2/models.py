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


class Net_Custom(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=1, activation=torch.nn.ReLU):
        super(Net_Custom, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)])
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.activation = activation()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        for layer in self.hidden_layers:
            out = layer(out)
            out = self.activation(out)
        out = self.fc2(out)
        return out

