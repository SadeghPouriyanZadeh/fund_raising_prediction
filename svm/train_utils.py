from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def train_with_kfold(x, y, n_splits):
    kf = KFold(n_splits, shuffle=True)
    scaler_y = StandardScaler()
    mean_val_loss = 0
    kfold_errors = []
    for train_index, test_index in kf.split(x):
        x_train = x[train_index, :]
        x_test = x[test_index, :]
        y_train = y[train_index, :]
        y_test = y[test_index, :]
        regressor = SVR(kernel="rbf")
        regressor.fit(
            x_train,
            y_train.ravel(),
        )
        pred = regressor.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        error = mean_squared_error(y_test, pred) ** 0.5
        mean_val_loss += error / n_splits
        kfold_errors.append(error)
    return mean_val_loss, kfold_errors


def train_svm(x, y, test_size=0.2, **best_params):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=1,
    )
    regressor = SVR(kernel="rbf", **best_params)
    regressor = regressor.fit(x_train, y_train)
    pred = regressor.predict(x_test)
    error = mean_squared_error(y_test, pred) ** 0.5
    y_pred = regressor.predict(x)
    return error, y_pred
