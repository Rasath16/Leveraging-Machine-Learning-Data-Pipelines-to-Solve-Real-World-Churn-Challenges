from __future__ import annotations
import pandas as pd
from typing import Dict, List

class FeatureEngineering:
    def __init__(self, service_cols: List[str], autopay_keywords: List[str]):
        self.service_cols = service_cols
        self.autopay_keywords = autopay_keywords

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        for col in self.service_cols:
            if col not in df.columns:
                
                df[col] = "No"
        df["ServiceAdoptionScore"] = df[self.service_cols].apply(
            lambda row: sum(val == "Yes" for val in row), axis=1
        )

        num_services = df["ServiceAdoptionScore"].replace(0, 1)
        if "MonthlyCharges" in df.columns:
            df["AvgChargesPerService"] = df["MonthlyCharges"] / num_services

        if "PaymentMethod" in df.columns:
            df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
            pattern = "|".join(self.autopay_keywords)
            df["IsAutoPay"] = df["PaymentMethod"].str.contains(pattern, case=False, na=False).astype(int)
        return df
