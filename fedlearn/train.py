import torch


def train(train_loader, model, criterion, optimizer, scheduler, device):
    batches = len(train_loader)
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
        # reset grad
        optimizer.zero_grad()
        # each source domain do optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / batches
    scheduler.step()
    return running_loss


def test(test_loader, model, criterion, device):
    correct = 0
    total = 0
    batches = len(test_loader)
    running_loss = 0.0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item() / batches
    return correct / total, running_loss
