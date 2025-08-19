from __future__ import annotations
import logging
import pandas as pd
from typing import Dict, List

class DataIngestion:
    def __init__(self, config: Dict):
        self.file_path: str = config["data"]["file_path"]
        self.drop_columns: List[str] = config["data"].get("drop_columns", [])
        self.target_column: str = config["data"]["target_column"]
        self.target_mapping: Dict = config["data"].get("target_mapping", {})

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            logging.info(f"Loaded data from {self.file_path} with shape {df.shape}")
            return df
        except Exception as e:
            logging.exception(f"Failed to load data from {self.file_path}")
            raise

    def basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for col in self.drop_columns:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                logging.info(f"Dropped column: {col}")

        # Map target labels
        if self.target_mapping:
            if self.target_column in df.columns:
                logging.info(f"Unique values before mapping: {df[self.target_column].unique()}")
                logging.info(f"Mapping being used: {self.target_mapping}")
                df[self.target_column] = df[self.target_column].str.strip().map(self.target_mapping)
                logging.info(f"Unique values after mapping: {df[self.target_column].unique()}")
                if df[self.target_column].isna().any():
                    nan_examples = df[df[self.target_column].isna()][self.target_column].head()
                    raise ValueError(
                        f"Target mapping introduced NaNs. Check mapping for column '{self.target_column}'. "
                        f"First few problematic values: {nan_examples}"
                    )
                logging.info(f"Mapped target column '{self.target_column}' using provided mapping.")
            else:
                raise KeyError(f"Target column '{self.target_column}' not found in data.")
        return df

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
