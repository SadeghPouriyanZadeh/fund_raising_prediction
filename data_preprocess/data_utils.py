from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import Normalizer, OneHotEncoder, OrdinalEncoder
from torch.utils.data import Dataset


def get_processed_data(ico_csv_path, normalize=True, one_hot_encode=True):
    df = pd.read_csv(ico_csv_path)
    df_x = df.drop(columns=["Total amount raised (USDm)"])
    df_y = df["Total amount raised (USDm)"]
    cat_cols = []
    con_cols = []
    for col in df_x.columns:
        if df_x[col].dtype == np.object_:
            cat_cols.append(col)
        else:
            con_cols.append(col)
    if one_hot_encode:
        cats = OneHotEncoder(sparse=False).fit_transform(df_x[cat_cols])
    else:
        cats = OrdinalEncoder().fit_transform(df_x[cat_cols])
    if normalize:
        cons = Normalizer().fit_transform(df_x[con_cols])
    else:
        cons = df_x[con_cols].to_numpy()
    x = np.concatenate((cats, cons), axis=1)
    y = df_y.to_numpy()[..., None]
    return x, y


class IcoDataset(Dataset):
    def __init__(self, x_ndarray, y_ndarray):
        self.x_ndarray = x_ndarray.astype(np.float32)
        self.y_ndarray = y_ndarray.astype(np.float32)
        self._length = self.x_ndarray.shape[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.x_ndarray[idx, :], self.y_ndarray[idx, :]

def get_cleaned_ico_df(df, max_allowed_col_nan):
    important_columns = [
        "total_number_of_tokens",
        "Whitepaper page count",
        "Smart contract code available",
        "Token share team (ex ante)",
        "Token share producers/miners (ex ante)",
        "Total amount raised (USDm)",
        "Crowdsale max. discount (%)",
        "price_crowdsale",
        "Length of crowdsale (calendar days, planned)",
    ]
    nans = df.isna()
    cols_nan_fraction = nans.sum(axis=0) / nans.shape[0]
    low_nan_cols = cols_nan_fraction[cols_nan_fraction < max_allowed_col_nan].index
    valid_cols = set(low_nan_cols.tolist() + important_columns)
    cleaned_df = df[valid_cols].dropna()
    return cleaned_df