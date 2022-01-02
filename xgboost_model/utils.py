from itertools import product
from pathlib import Path

import numpy as np
from dataset.data import get_processed_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from xgboost import XGBRegressor


def param_generator():
    params_values = {
        "n_estimators": [100, 1000],
        "learning_rate": np.random.uniform(0.01, 0.1, 2),  #
        "min_split_loss": np.random.randint(
            0, 10, 2
        ),  # The larger gamma is, the more conservative the algorithm will be
        "max_depth": np.random.randint(
            3, 11, 2
        ),  # Increasing this value will make the model more complex and more likely to overfit.
        "min_child_weight": np.random.uniform(
            0.0, 10, 2
        ),  # Too high values can lead to under-fitting.
        "subsample": np.random.uniform(
            0.75, 1, 2
        ),  # Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
        "colsample_bytree": np.random.uniform(0.5, 1, 2),
        "colsample_bylevel": np.random.uniform(0.5, 1, 2),
        "colsample_bynode": np.random.uniform(0.5, 1, 2),
        "reg_lambda": np.random.uniform(
            1, 10, 2
        ),  # Increasing this value will make model more conservative.
        "reg_alpha": np.random.uniform(
            0, 1, 3
        ),  # Increasing this value will make model more conservative.
        # "tree_method": ["gpu_hist",],
        "random_state": [
            0,
        ],
    }
    keys = params_values.keys()
    for row in product(*[params_values[key] for key in keys]):
        data = {}
        for i, key in enumerate(keys):
            data[key] = row[i]
        yield data


def train_with_kfold(x, y, params):
    kf5 = KFold(n_splits=5, shuffle=True)
    mean_val_loss = 0
    for train_index, test_index in kf5.split(x):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index, :]
        y_test = y[test_index, :]
        evaluation = [(x_train, y_train), (x_test, y_test)]
        model = XGBRegressor(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=evaluation,
            eval_metric="rmse",
            early_stopping_rounds=10,
            verbose=False,
        )
        pred = model.predict(x_test)
        error = mean_squared_error(y_test, pred) ** 0.5
        mean_val_loss += error / 5
    return mean_val_loss


def find_best_params(params_gen=param_generator()):
    min_error = float("inf")
    best_params = None
    for params in params_gen:
        val_loss = train_with_kfold(x, y, params)
        if val_loss < min_error:
            min_error = val_loss
            best_params = params
            print("Found new min error:", min_error)
            print(f"{best_params}=")
            print("=" * 88)
    return best_params, min_error
