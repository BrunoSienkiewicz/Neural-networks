import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


def train_model(model, train, valid, criterion, optimizer, eval_func, batch_size=100, num_epochs=1000, verbose=True, epoch_interval=100, device=torch.device('cuda')):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    iters_list = []
    loss_list = []
    train_eval_list = []
    valid_eval_list = []

    _range = range(num_epochs)
    if verbose:
        _range = tqdm(_range)
    for epoch in _range:
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            model.train()
            y_pred = model(x)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % epoch_interval == 0:
            iters_list.append(epoch)
            loss_list.append(loss.item())
            train_eval_list.append(eval_func(model, train))
            valid_eval_list.append(eval_func(model, valid))

    return iters_list, loss_list, train_eval_list, valid_eval_list


def plot_training(iters, loss, train_eval, valid_eval, ax=None):

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


def get_confusion_matrix(model, data, device=torch.device('cuda')):
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        y_pred = torch.round(y_pred)
        y = y.view_as(y_pred)
        confusion_matrix = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                confusion_matrix[i, j] = torch.sum((y == i) & (y_pred == j))
    model.train()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, ax=ax, fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.suptitle("Confusion matrix")
    fig.tight_layout()
    plt.show()


def get_mse(model, data, device=torch.device('cuda')):
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        mse = torch.mean((y_pred - y) ** 2)
    model.train()
    return mse.item()

def get_accuracy(model, data, device=torch.device('cuda')):
    correct = 0
    total = 0
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        y_pred = torch.round(output)
        correct = y_pred.eq(y.view_as(y_pred)).sum().item()
        total = y_pred.shape[0]
    model.train()
    if ((correct / total)>0.20):
      return max(correct / total)
    else:
      return get_accuracy_regression(mmodel,data)


def get_accuracy_regression(model, data):
    correct = 0
    total = 0
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        output = model(x)
        for prediction, true_value in zip(output, y):
            print(true_value, prediction)

            if true_value <=100000 and 100000 >= prediction :
                correct += 1
            elif true_value>100000 and true_value<350000 and 100000 <= prediction and prediction<350000:
                correct += 1
            elif true_value>=350000 and 350000 >= prediction :
                correct += 1
        total += y.shape[0]
    return correct / total



def get_overfitting(model, train_data, valid_data, eval_func):
    train_eval = eval_func(model, train_data)
    valid_eval = eval_func(model, valid_data)
    return train_eval - valid_eval
