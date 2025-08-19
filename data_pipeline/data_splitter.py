from __future__ import annotations
from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

class DataSplitter:
    @staticmethod
    def split(X: pd.DataFrame, y, test_size: float, random_state: int):
        return train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)
