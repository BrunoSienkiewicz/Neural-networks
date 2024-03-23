import torch


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=torch.device('cuda')):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self = self.to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class NetLayers(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, device=torch.device('cuda')):
        super(NetLayers, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        for i in range(num_layers-1):
            setattr(self, f'fc{i+2}', torch.nn.Linear(hidden_size, hidden_size))
            setattr(self, f'relu_{i+2}', torch.nn.ReLU())
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self = self.to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for i in range(2, 3):
            out = getattr(self, f'fc{i}')(out)
            out = getattr(self, f'relu_{i}')(out)
        out = self.fc3(out)
        return out


class NetBatchNorm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device=torch.device('cuda')):
        super(NetBatchNorm, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.relu_2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self = self.to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.fc3(out)
        return out
    

class NetDropout(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5, device=torch.device('cuda')):
        super(NetDropout, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu_1 = torch.nn.ReLU()
        self.d1 = torch.nn.Dropout(p=dropout_p)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu_2 = torch.nn.ReLU()
        self.d2 = torch.nn.Dropout(p=dropout_p)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self = self.to(device)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu_1(out)
        out = self.d1(out)
        out = self.fc2(out)
        out = self.relu_2(out)
        out = self.d2(out)
        out = self.fc3(out)
        return out


class NetEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, device=torch.device('cuda')):
        super(NetEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self = self.to(device)

    def forward(self, x):
        x = x.long()
        out = self.embedding(x)
        out = out.mean(dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out