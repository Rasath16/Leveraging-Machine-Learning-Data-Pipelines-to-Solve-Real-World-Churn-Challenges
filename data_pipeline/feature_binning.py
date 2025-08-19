from __future__ import annotations
import pandas as pd
from typing import List

class FeatureBinning:
    def __init__(self, tenure_bins: List[float], tenure_labels: List[str]):
        self.tenure_bins = tenure_bins
        self.tenure_labels = tenure_labels

    def add_tenure_category(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tenure" in df.columns:
            df["TenureCategory"] = pd.cut(
                df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=False
            )
        return df
