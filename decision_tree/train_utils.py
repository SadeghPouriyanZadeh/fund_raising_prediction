# calculating different regression metrics

from pathlib import Path

from data_preprocess.data_utils import get_processed_data
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split


def find_best_hyperparameter(
    data_path,
    regressor,
    target_feature,
    parameters: dict,
    scoring="neg_mean_squared_error",
):

    x, y = get_processed_data(data_path, target_feature)
    tuning_model = GridSearchCV(
        regressor, param_grid=parameters, scoring=scoring, cv=5, verbose=1
    )
    tuning_model.fit(x, y)
    return tuning_model.best_params_


def train_tree_with_kfold(
    data_path,
    target_feature,
    kflod_n_splits,
    kfold_shuffle=True,
    **best_params,
):
    x, y = get_processed_data(data_path, target_feature)
    regressor = tree.DecisionTreeRegressor(**best_params)
    kf = KFold(n_splits=kflod_n_splits, shuffle=kfold_shuffle)
    error = 0
    fold_errors = []
    for train_index, test_index in kf.split(x, y):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index, :]
        y_test = y[test_index, :]

        regressor = regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        fold_error = mean_squared_error(y_test, pred) ** 0.5
        error += fold_error / kflod_n_splits
        fold_errors.append(fold_error)
    return error, fold_errors


def train_tree(x, y, test_size=0.2, **best_params):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=1,
    )
    regressor = tree.DecisionTreeRegressor(**best_params)
    regressor = regressor.fit(x_train, y_train)
    pred = regressor.predict(x_test)
    error = mean_squared_error(y_test, pred) ** 0.5
    y_pred = regressor.predict(x)
    return error, y_pred
