from itertools import product

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data import IcoDataset, get_processed_data
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


def param_generator():
    params_values = {
        "batch_size": [2 ** i for i in range(5, 7)],
        "hidden_layers": [2 ** i for i in range(2, 4)],
        "layer_units": [2 ** i for i in range(4, 7)],
        "learning_rate": np.random.uniform(1e-5, 0.1, 3).tolist(),
        "normalize": [True, False],
        "one_hot_encode": [True, False],
    }
    keys = [
        "batch_size",
        "hidden_layers",
        "layer_units",
        "learning_rate",
        "normalize",
        "one_hot_encode",
    ]
    for row in product(*[params_values[key] for key in keys]):
        data = {}
        for i, key in enumerate(keys):
            data[key] = row[i]
        yield data


class HyperParameterLogger:
    def __init__(
        self,
        csv_file_path,
    ):
        self.csv_file_path = csv_file_path
        self.history = {
            "batch_size": [],
            "hidden_layers": [],
            "layer_units": [],
            "learning_rate": [],
            "normalize": [],
            "one_hot_encode": [],
            "epochs": [],
            "data_path": [],
            "device": [],
            "val_loss": [],
            "train_time": [],
        }

    def log(self, param_dict, val_loss, train_time):
        self.history["val_loss"].append(val_loss)
        self.history["train_time"].append(train_time)
        for key in param_dict:
            self.history[key].append(param_dict[key])
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_file_path, index=False)
        return df


def find_best_hyperparameter(df: DataFrame, val_loss_label: str):
    min_val_loss = min(df[val_loss_label])
    min_val_loss_idx = np.where(df[val_loss_label] == min_val_loss)
    hyper_parameters = df.values[min_val_loss_idx]
    return hyper_parameters
