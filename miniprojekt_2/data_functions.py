import torch
import torch.utils.data
import pandas as pd
import numpy as np


def train_valid_split(data, target, val_ratio=0.2) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
    train = data.sample(frac=1 - val_ratio, random_state=200)
    val = data.drop(train.index)

    train_x = np.array(train.drop(target, axis=1).values, dtype=np.float32)
    train_y = np.array(train[target].values, dtype=np.float32)
    val_x = np.array(val.drop(target, axis=1).values, dtype=np.float32)
    val_y = np.array(val[target].values, dtype=np.float32)

    train_torch_x = torch.from_numpy(train_x).clone().detach().float()
    train_torch_y = torch.from_numpy(train_y).clone().detach().float()
    val_torch_x = torch.from_numpy(val_x).clone().detach().float()
    val_torch_y = torch.from_numpy(val_y).clone().detach().float()
    train_dataset = torch.utils.data.TensorDataset(train_torch_x, train_torch_y)
    val_dataset = torch.utils.data.TensorDataset(val_torch_x, val_torch_y)

    return train_dataset, val_dataset


def get_dummies(data, columns):
    cat_values = data[columns]
    cat_values = pd.get_dummies(cat_values)
    data = data.drop(columns, axis=1)
    data = pd.concat([data, cat_values], axis=1)
    return data