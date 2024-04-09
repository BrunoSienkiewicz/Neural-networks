import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, criterion, optimizer, eval_fn, device, num_epochs=5):
    train_eval_hist = []
    val_eval_hist = []
    loss_hist = []
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss_hist.append(loss.item())
        train_eval_hist.append(eval_fn(model, train_loader, device))
        val_eval_hist.append(eval_fn(model, val_loader, device))
    return loss_hist, train_eval_hist, val_eval_hist
         
            
def get_accuracy(model, data, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
    return correct / total