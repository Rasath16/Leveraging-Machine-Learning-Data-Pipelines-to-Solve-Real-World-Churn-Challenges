import pytest
import pandas as pd
import numpy as np
from data_pipeline.data_ingestion import DataIngestion

@pytest.fixture
def sample_config():
    return {
        "data": {
            "file_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "drop_columns": ["customerID"],
            "target_column": "Churn",
            "target_mapping": {"Yes": 1, "No": 0}
        }
    }

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'Churn': ['Yes', 'No', 'Yes'],
        'MonthlyCharges': [70.35, 55.50, 88.40],
        'TotalCharges': [1000.20, 800.30, 1500.50]
    })

def test_data_ingestion_initialization(sample_config):
    ingestion = DataIngestion(sample_config)
    assert ingestion.file_path == sample_config["data"]["file_path"]
    assert ingestion.drop_columns == sample_config["data"]["drop_columns"]
    assert ingestion.target_column == sample_config["data"]["target_column"]
    assert ingestion.target_mapping == sample_config["data"]["target_mapping"]

def test_basic_clean(sample_config, sample_df):
    ingestion = DataIngestion(sample_config)
    cleaned_df = ingestion.basic_clean(sample_df.copy())
    
    # Test column dropping
    assert "customerID" not in cleaned_df.columns
    
    # Test target mapping
    assert cleaned_df["Churn"].dtype == np.int64
    assert set(cleaned_df["Churn"].unique()) == {0, 1}

def test_validate_columns():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    required_cols = ['A', 'B']
    
   
    DataIngestion.validate_columns(df, required_cols)
    
    
    with pytest.raises(ValueError):
        DataIngestion.validate_columns(df, ['A', 'C'])
