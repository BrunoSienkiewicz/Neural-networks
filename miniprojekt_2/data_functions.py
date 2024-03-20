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


def train_valid_split(data, val_ratio=0.2):
    train = data.sample(frac=1 - val_ratio, random_state=200)
    val = data.drop(train.index)
    return train, val


def data_to_dataset(data, target) -> torch.utils.data.Dataset:
    x = np.array(data.drop(target, axis=1).values, dtype=np.float32)
    y = np.array(data[target].values, dtype=np.float32)

    torch_x = torch.from_numpy(x).clone().detach().float()
    torch_y = torch.from_numpy(y).clone().detach().float()
    dataset = torch.utils.data.TensorDataset(torch_x, torch_y)

    return dataset


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

