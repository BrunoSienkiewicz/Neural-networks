import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, data, labels):
        self.data = []
        for d, l in zip(data,labels):
            self.data.append((torch.from_numpy(np.array(d)).long(),torch.tensor(l).long()))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        return in_data, target
    
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