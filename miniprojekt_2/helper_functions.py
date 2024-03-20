import torch
import matplotlib.pyplot as plt

from tqdm import tqdm


def train_model(model, train, valid, criterion, optimizer, eval_func, batch_size=100, num_epochs=1000, verbose=True, epoch_interval=100):
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


def plot_training(iters, loss, train_eval, valid_eval):

    fig, ax1 = plt.subplots()

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

    fig.tight_layout()
    plt.show()


def get_mse(model, data):
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        y_pred = model(x)
        mse = torch.mean((y_pred - y) ** 2)
    return mse.item()

def get_accuracy(model, data):
    correct = 0
    total = 0
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        output = model(x)
        y_pred = torch.round(output)
        correct = y_pred.eq(y.view_as(y_pred)).sum().item()
        total = y_pred.shape[0]
    return correct / total
