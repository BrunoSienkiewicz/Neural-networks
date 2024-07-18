import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from torch.nn.utils.rnn import pad_sequence 


def get_accuracy(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _, _ in data_loader:
            x, y = x.to(device).unsqueeze(2), y.to(device)
            hidden, state = net.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)
            preds, _ = net(x, (hidden, state))
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    net.train()
    return correct / total

def get_accuracy_features(net, data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, f, y, _, _, _ in data_loader:
            x, f, y = x.to(device).unsqueeze(2), f.to(device), y.to(device)
            hidden, state = net.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)
            preds, _ = net(x, (hidden, state), f)
            _, predicted = torch.max(preds, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    net.train()
    return correct / total

def get_confusion_matrix(net, data_loader, n_classes, device):
    net.eval()
    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for x, y, _, _ in data_loader:
            x, y = x.to(device).unsqueeze(2), y.to(device)
            hidden, state = net.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)
            preds, _ = net(x, (hidden, state))
            _, predicted = torch.max(preds, 1)
            for t, p in zip(y.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    net.train()
    return confusion_matrix

def get_confusion_matrix_features(net, data_loader, n_classes, device):
    net.eval()
    confusion_matrix = torch.zeros(n_classes, n_classes)
    with torch.no_grad():
        for x, f, y, _, _, _ in data_loader:
            x, f, y = x.to(device).unsqueeze(2), f.to(device), y.to(device)
            hidden, state = net.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)
            preds, _ = net(x, (hidden, state), f)
            _, predicted = torch.max(preds, 1)
            for t, p in zip(y.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    net.train()
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    confusion_matrix = confusion_matrix.float() / confusion_matrix.sum(1).view(-1, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.size(0)):
        for j in range(confusion_matrix.size(1)):
            plt.text(j, i, format(confusion_matrix[i, j], ".2f"),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_training(loss, train_eval, valid_eval, ax=None):
    iters = np.arange(len(loss))
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iters, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Evaluation', color=color)
    ax2.plot(iters, train_eval, color=color, linestyle='dashed', label='Train')
    ax2.plot(iters, valid_eval, color=color, linestyle='solid', label='Validation')
    ax2.tick_params(axis='y', labelcolor=color)

    if ax is None:
        fig.tight_layout()
        plt.legend()
        plt.show()


def pad_collate(batch, pad_value=-1):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [1 for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy = torch.stack(yy)

    return xx_pad, yy, x_lens, y_lens

def pad_collate_feat(batch, pad_value=-1):
    xx, ff, yy = zip(*batch)

    x_lens = []
    for x in xx:
        try:
            x_lens.append(len(x))
        except:
            print(x)
            raise
    f_lens = [len(f) for f in ff]  
    y_lens = [1 for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    ff = torch.stack(ff)
    yy = torch.stack(yy)

    return xx_pad, ff, yy, x_lens, f_lens, y_lens