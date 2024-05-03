import torch
            
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

    
def get_balanced_accuracy(model, data, classes, device):
    model.eval()
    # calculate accuracy for each class
    class_correct = torch.zeros(len(classes))
    class_total = torch.zeros(len(classes))
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            for i in range(len(images)):
                class_correct[labels[i]] += int(predicted[i] == labels[i])
                class_total[labels[i]] += 1
    class_acc = class_correct / class_total
    return class_acc.mean().item(), class_acc

    
def get_confusion_matrix(model, data, classes, device):
    model.eval()
    confusion_matrix = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            for i in range(len(images)):
                confusion_matrix[labels[i]][predicted[i]] += 1
    return confusion_matrix