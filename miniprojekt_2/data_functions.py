import torch
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE


labels = {
    "cheap": lambda x: x <= 100_000,
    "average": lambda x: x > 100_000 and x <= 350_000,
    "expensive": lambda x: x > 350_000
}

labels_to_num = {
    "cheap": 0,
    "average": 1,
    "expensive": 2
}


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

def output_to_labels(data, target):
    data[target+"_label"] = data[target].apply(lambda x: "cheap" if labels["cheap"](x) else "average" if labels["average"](x) else "expensive")
    data = data.drop(target, axis=1)
    data.rename(columns={target+"_label": target}, inplace=True)
    data[target] = data[target].apply(lambda x: labels_to_num[x])
    return data

def equalize_classes(df, target):
    smote = SMOTE()
    x = df.drop(target, axis=1)
    y = df[target]
    x_smote, y_smote = smote.fit_resample(x, y)
    df = pd.concat([x_smote, y_smote], axis=1)
    return df

