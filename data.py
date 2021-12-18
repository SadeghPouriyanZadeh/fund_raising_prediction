from pathlib import Path

from pandas import DataFrame
from sklearn.preprocessing import Normalizer, OneHotEncoder, OrdinalEncoder
from torch.utils.data import Dataset
import numpy as np
from data_processing import get_processed_data


class IcoDataset(Dataset):
    def __init__(self, x_ndarray, y_ndarray):
        self.x_ndarray = x_ndarray.astype(np.float32)
        self.y_ndarray = y_ndarray.astype(np.float32)
        self._length = self.x_ndarray.shape[0]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.x_ndarray[idx, :], self.y_ndarray[idx, :]
