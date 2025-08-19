from __future__ import annotations
from typing import Tuple
from imblearn.over_sampling import SMOTE

class ImbalanceHandler:
    def __init__(self, random_state: int = 42):
        self.smote = SMOTE(random_state=random_state)

    def fit_resample(self, X, y):
        return self.smote.fit_resample(X, y)
