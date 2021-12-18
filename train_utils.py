import torch

def train_loop(model, dataloader, criterion, optimizer):
    mean_loss = 0
    for x, y in dataloader:
        pred = model(x)
        loss = criterion(pred.squeeze(), y)
        mean_loss += loss.item() / len(dataloader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean_loss


@torch.no_grad()
def val_loop(model, dataloader, criterion):
    mean_loss = 0
    for x, y in dataloader:
        pred = model(x)
        loss = criterion(pred.squeeze(), y)
        mean_loss += loss.item() / len(dataloader)
    return mean_loss


def train(epochs, model, dataloader, criterion, optimizer):
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        train_loss = train_loop(model, dataloader, criterion, optimizer)
        val_loss = val_loop(model, dataloader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses
