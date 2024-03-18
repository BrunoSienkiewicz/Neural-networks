import torch

from tqdm import tqdm


def train_model(model, train, valid, criterion, optimizer, eval_func, batch_size=100, num_epochs=1000, learn_rate=0.001):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    iters_list = []
    loss_list = []
    train_eval_list = []
    valid_eval_list = []
    for epoch in tqdm(range(num_epochs)):
        for i, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 100 == 0:
            iters_list.append(epoch)
            loss_list.append(loss.item())
            train_eval_list.append(eval_func(model, train))
            valid_eval_list.append(eval_func(model, valid))

    return iters_list, loss_list, train_eval_list, valid_eval_list


def get_mse(model, data):
    model.eval()
    for x, y in torch.utils.data.DataLoader(data, batch_size=len(data)):
        y_pred = model(x)
        mse = torch.mean((y_pred - y) ** 2)
    return mse.item()
