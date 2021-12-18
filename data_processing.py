import numpy as np
from sklearn.preprocessing import Normalizer, OneHotEncoder, OrdinalEncoder
import pandas as pd

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
