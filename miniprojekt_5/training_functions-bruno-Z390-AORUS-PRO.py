import torch
from tqdm import tqdm


def train_lstm(model, train_loader, val_loader, optimizer, loss_fn, eval_fn, n_epochs=100, verbose=True, device='cuda'):
    if verbose:
        rng = tqdm(range(n_epochs))
    else:
        rng = range(n_epochs)

    loss_hist = []
    train_eval_hist = []
    val_eval_hist = []

    for epoch in rng:
        for x, targets, _, _ in train_loader:
            x = x.to(device).unsqueeze(2)
            targets = targets.to(device)
            hidden, state = model.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device) 
            preds, _ = model(x, (hidden,state))
            targets = targets.squeeze()
            targets = targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad() 
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 and verbose:
            print(f"Epoch: {epoch}, loss: {loss.item():.3}")

        loss_hist.append(loss.item())
        train_eval_hist.append(eval_fn(model, train_loader, device))
        val_eval_hist.append(eval_fn(model, val_loader, device))

    return loss_hist, train_eval_hist, val_eval_hist

def train_lstm_features(model, train_loader, val_loader, optimizer, loss_fn, eval_fn, n_epochs=100, verbose=True, device='cuda'):
    if verbose:
        rng = tqdm(range(n_epochs))
    else:
        rng = range(n_epochs)

    loss_hist = []
    train_eval_hist = []
    val_eval_hist = []

    for epoch in rng:
        for x, features, targets, _, _, _ in train_loader:
            x = x.to(device).unsqueeze(2)
            features = features.to(device)
            targets = targets.to(device)
            hidden, state = model.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device) 
            preds, _ = model(x, (hidden,state), features)
            targets = targets.squeeze()
            targets = targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad() 
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 and verbose:
            print(f"Epoch: {epoch}, loss: {loss.item():.3}")

        loss_hist.append(loss.item())
        train_eval_hist.append(eval_fn(model, train_loader, device))
        val_eval_hist.append(eval_fn(model, val_loader, device))

    return loss_hist, train_eval_hist, val_eval_hist
