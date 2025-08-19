from __future__ import annotations
import logging
from typing import Dict
import numpy as np
import pandas as pd
from scipy import sparse

from data_pipeline.data_ingestion import DataIngestion
from data_pipeline.handle_missing_values import MissingValueHandler
from data_pipeline.feature_binning import FeatureBinning
from data_pipeline.feature_engineering import FeatureEngineering
from data_pipeline.data_splitter import DataSplitter
from data_pipeline.feature_encoding import PreprocessorFactory
from data_pipeline.imbalance_handler import ImbalanceHandler
from data_pipeline.artifact_saver import ArtifactSaver

class DataPipeline:
    def __init__(self, config: Dict):
        self.config = config

    def run(self) -> None:
      
        ingestion = DataIngestion(self.config)
        df = ingestion.load_data()
        df = ingestion.basic_clean(df)

        
        mv = MissingValueHandler(
            strategy=self.config["preprocessing"]["missing_value_strategy"],
            numeric_to_coerce=self.config["preprocessing"].get("numeric_to_coerce", []),
        )
        df = mv.coerce_and_impute(df)

     
        bin_conf = self.config["preprocessing"]["binning"]
        df = FeatureBinning(bin_conf["tenure_bins"], bin_conf["tenure_labels"]).add_tenure_category(df)

        fe = FeatureEngineering(
            service_cols=self.config["preprocessing"]["service_columns"],
            autopay_keywords=self.config["preprocessing"]["autopay_keywords"],
        )
        df = fe.add_features(df)


        target = self.config["data"]["target_column"]
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = DataSplitter.split(
            X, y,
            test_size=self.config["preprocessing"]["test_size"],
            random_state=self.config["preprocessing"]["random_state"],
        )

      
        numeric_features = list(X.select_dtypes(include=["int64", "float64"]).columns)
        categorical_features = list(X.select_dtypes(include=["object", "category"]).columns)
        preprocessor = PreprocessorFactory.create(numeric_features, categorical_features, self.config)

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)


        if sparse.issparse(X_train_proc):
            X_train_proc = X_train_proc.toarray()
        if sparse.issparse(X_test_proc):
            X_test_proc = X_test_proc.toarray()

        
        if self.config["preprocessing"].get("smote", False):
            smote = ImbalanceHandler(random_state=self.config["preprocessing"]["random_state"])
            X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)

   
        feature_names = PreprocessorFactory.get_feature_names(preprocessor, numeric_features, categorical_features)

        ArtifactSaver.save_npz(self.config["artifacts"]["x_train"], X_train_proc)
        ArtifactSaver.save_npz(self.config["artifacts"]["y_train"], y_train.values if hasattr(y_train, 'values') else y_train)
        ArtifactSaver.save_npz(self.config["artifacts"]["x_test"], X_test_proc)
        ArtifactSaver.save_npz(self.config["artifacts"]["y_test"], y_test.values if hasattr(y_test, 'values') else y_test)

        ArtifactSaver.save_npy(self.config["artifacts"]["feature_names"], np.array(feature_names, dtype=object))
        ArtifactSaver.save_preprocessor(self.config["artifacts"]["preprocessor"], preprocessor)

        logging.info("Data pipeline completed successfully.")
