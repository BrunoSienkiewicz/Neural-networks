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
    correct_labels = {i : 0 for i in range(len(classes))}
    total_labels = {i : 0 for i in range(len(classes))}
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            for i in range(len(classes)):
                correct_labels[i] += int((predicted == labels).sum())
                total_labels[i] += int((labels == i).sum())
    balanced_accuracy = {i : correct_labels[i]/total_labels[i] for i in range(len(classes))}
    return balanced_accuracy, sum(balanced_accuracy.values()) / len(balanced_accuracy)

    
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