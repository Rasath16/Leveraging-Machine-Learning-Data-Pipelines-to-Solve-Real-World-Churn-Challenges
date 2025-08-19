from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    A safe label/ordinal encoder for multiple categorical columns.
    Uses OrdinalEncoder under the hood with unknowns -> -1, which emulates label encoding.
    """
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.encoder.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame):
        enc = self.encoder.transform(X[self.columns_])
        return enc

class PreprocessorFactory:
    @staticmethod
    def _scaler(kind: str):
        if kind == "minmax":
            return MinMaxScaler()
        return StandardScaler()

    @staticmethod
    def _categorical_transformer(kind: str):
        if kind == "onehot":
            # Return dense to make it compatible with later SMOTE
            return Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        elif kind in ("label", "ordinal"):
            return Pipeline([("encoder", LabelEncodingTransformer())])
        else:
            raise ValueError(f"Unknown encoding kind: {kind}")

    @staticmethod
    def create(numeric_features: List[str], categorical_features: List[str], config: Dict) -> ColumnTransformer:
        enc_kind = config["preprocessing"]["encoding"]
        sc_kind = config["preprocessing"]["scaling"]

        numeric_transformer = Pipeline([("scaler", PreprocessorFactory._scaler(sc_kind))])
        categorical_transformer = PreprocessorFactory._categorical_transformer(enc_kind)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        return preprocessor

    @staticmethod
    def get_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
        """Return output feature names depending on the encoder used."""
        cat_transformer = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        if isinstance(cat_transformer, OneHotEncoder):
            cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
        else:
            cat_names = list(categorical_features)
        return list(numeric_features) + cat_names
