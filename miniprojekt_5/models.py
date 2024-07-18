import torch
import torch.nn as nn


class LSTM_var_Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_var_Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def init_hidden(self, batch_size=1) -> tuple:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden):
        all_outputs, hidden = self.lstm(x, hidden)
        out = self.fc(all_outputs[:, -1, :])
        return out, hidden
    
class LSTM_features(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, num_classes, num_features, hidden_layers=1, dropout=0.1):
        super(LSTM_features, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)

        self.fc_in = nn.Linear(num_features, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(hidden_layers)]
        )

        self.fc_out = nn.Linear(hidden_size + hidden_size, num_classes)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def init_hidden(self, batch_size=1) -> tuple:
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_size),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden, features):
        all_outputs, hidden = self.lstm(x, hidden)

        features = self.LeakyReLU(self.fc_in(features))
        for layer in self.layers:
            features = self.LeakyReLU(layer(features))

        x = torch.cat((all_outputs[:, -1, :], features), dim=1)
        out = self.fc_out(x)
        return out, hidden
    
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, max_length, bidirectional=False):
        super(LSTMRegressor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, dropout=0.4)
        self.fc = nn.Linear(hidden_size * self.bidirectional * max_length, out_size)

    def init_hidden(self, batch_size):
        # Inicjalizacja stanów ukrytych
        hidden = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        x = x.transpose(0, 1)
        all_outputs, hidden = self.lstm(x, hidden)
        all_outputs = all_outputs.transpose(0, 1)
        out = all_outputs.contiguous().view(all_outputs.size(0), -1)
        x = self.fc(out)
        return x, hidden
    
class LSTMRegressorFeatures(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size, num_features, max_length, hidden_layers=1, dropout=0.1, bidirectional=False):
        super(LSTMRegressorFeatures, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)

        self.fc_in = nn.Linear(num_features, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(hidden_layers)]
        )

        self.fc_out = nn.Linear(hidden_size * self.bidirectional * max_length + hidden_size, out_size)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def init_hidden(self, batch_size):
        # Inicjalizacja stanów ukrytych
        hidden = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden, features):
        x = x.transpose(0, 1)
        all_outputs, hidden = self.lstm(x, hidden)

        features = self.LeakyReLU(self.fc_in(features))
        for layer in self.layers:
            features = self.LeakyReLU(layer(features))

        all_outputs = all_outputs.transpose(0, 1)
        x = torch.cat((all_outputs.contiguous().view(all_outputs.size(0), -1), features), dim=1)
        x = self.fc_out(x)
        return x, hidden