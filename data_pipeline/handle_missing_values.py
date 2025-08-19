from __future__ import annotations
import logging
import pandas as pd
from typing import Dict, List

class MissingValueHandler:
    def __init__(self, strategy: str, numeric_to_coerce: List[str]):
        self.strategy = strategy
        self.numeric_to_coerce = numeric_to_coerce

    def coerce_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for col in self.numeric_to_coerce:
            if col not in df.columns:
                logging.warning(f"Column '{col}' not in dataframe during coercion.")
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if self.strategy == "median":
                fill_val = df[col].median()
            elif self.strategy == "mean":
                fill_val = df[col].mean()
            else:
                fill_val = df[col].mode().iloc[0]
            na_before = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            logging.info(f"Imputed {na_before} NaNs in '{col}' using {self.strategy}.")
        return df
