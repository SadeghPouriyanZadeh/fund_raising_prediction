import time
from itertools import product

import numpy as np
import pandas as pd
from pandas import DataFrame

from neural_network.train_utils import run_with_kfold


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


def param_generator():
    params_values = {
        "batch_size": [2 ** i for i in range(5, 8)],
        "hidden_layers": [2 ** i for i in range(2, 5)],
        "layer_units": [2 ** i for i in range(4, 8)],
        "learning_rate": np.random.uniform(1e-5, 0.1, 5).tolist(),
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


def find_best_hyperparameter(df: DataFrame, val_loss_label: str):
    min_val_loss = min(df[val_loss_label])
    min_val_loss_idx = np.where(df[val_loss_label] == min_val_loss)
    hyper_parameters = df.values[min_val_loss_idx]
    return hyper_parameters, min_val_loss_idx


def tune_hyperparameters(data_path, log_path, epochs, device, params):
    logger = HyperParameterLogger(log_path)
    for param_dict in params:
        param_dict["epochs"] = epochs
        param_dict["data_path"] = data_path
        param_dict["device"] = device
        print(param_dict)
        tic = time.time()
        val_loss = run_with_kfold(**param_dict)
        toc = time.time()
        print("validation loss :", val_loss)
        print("time length for one fold :", toc - tic)
        logger.log(param_dict, val_loss, toc - tic)
        print("=" * 80)
