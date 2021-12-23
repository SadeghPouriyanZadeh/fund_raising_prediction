import numpy as np
import torch
from dataset.data import IcoDataset, get_processed_data
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from neural_network.model import IcoPredictor


def train_loop(model, dataloader, criterion, optimizer, device):
    mean_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        mean_loss += loss.item() / len(dataloader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean_loss


@torch.no_grad()
def val_loop(model, dataloader, criterion, device):
    mean_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        mean_loss += loss.item() / len(dataloader)
    return mean_loss


def train(
    epochs, model, train_dataloader, valid_dataloader, criterion, optimizer, device
):
    train_losses = []
    val_losses = []
    for _ in range(epochs):
        train_loss = train_loop(model, train_dataloader, criterion, optimizer, device)
        val_loss = val_loop(model, valid_dataloader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return np.array(train_losses), np.array(val_losses)


def run_with_kfold(
    data_path,
    epochs,
    device,
    batch_size,
    hidden_layers,
    layer_units,
    learning_rate,
    normalize,
    one_hot_encode,
    **kwargs,
):
    x, y = get_processed_data(
        data_path, normalize=normalize, one_hot_encode=one_hot_encode
    )
    kf5 = KFold(n_splits=5, shuffle=True)
    total_val_losses = []
    for train_index, test_index in kf5.split(x):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index, :]
        y_test = y[test_index, :]
        train_dataset = IcoDataset(x_train, y_train)
        test_dataset = IcoDataset(x_test, y_test)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        model = IcoPredictor(
            x.shape[1], hidden_layers=hidden_layers, layer_units=layer_units
        ).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, val_losses = train(
            epochs,
            model,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            device,
        )
        total_val_losses.append(min(val_losses))
    val_loss = np.sqrt(np.array(total_val_losses).mean())
    return val_loss
