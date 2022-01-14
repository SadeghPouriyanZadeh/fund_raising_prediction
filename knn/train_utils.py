from data_preprocess.data_utils import get_processed_data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor


def train_knn_with_kfold(
    data_path,target_feature, kflod_n_splits, kfold_shuffle=True, **best_params
):
    x, y = get_processed_data(data_path, target_feature=target_feature)
    regressor = KNeighborsRegressor(**best_params)
    kf = KFold(n_splits=kflod_n_splits, shuffle=kfold_shuffle)
    error = 0
    kfold_errors = []
    for train_index, test_index in kf.split(x, y):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index, :]
        y_test = y[test_index, :]

        regressor = regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        fold_error = mean_squared_error(y_test, pred) ** 0.5
        error += fold_error / kflod_n_splits
        kfold_errors.append(fold_error)
    return error, kfold_errors


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
