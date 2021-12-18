from torch.utils.data import Dataset
from pandas import DataFrame
import numpy as np

class CancerDataset(Dataset):
    def __init__(self, df: DataFrame):
        self.df_x = df.drop(columns=["lung_cancer"])
        self.df_y = df["lung_cancer"]
        self._length = len(self.df_x)
    def __len__(self):
        return self._length
    def __getitem__(self, idx):
        x = self.df_x.iloc[idx].to_numpy().astype(np.float32)
        y = self.df_y.iloc[idx].astype(np.float32)
        return x, y