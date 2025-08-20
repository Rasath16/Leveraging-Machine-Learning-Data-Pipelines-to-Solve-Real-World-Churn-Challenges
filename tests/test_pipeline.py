import pytest
import pandas as pd
import numpy as np
from data_pipeline.data_pipeline import DataPipeline

@pytest.fixture
def sample_config():
    return {
        "data": {
            "file_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "drop_columns": ["customerID"],
            "target_column": "Churn",
            "target_mapping": {"Yes": 1, "No": 0}
        },
        "preprocessing": {
            "missing_value_strategy": "mean",
            "numeric_to_coerce": ["TotalCharges"],
            "test_size": 0.2,
            "service_columns": [
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ],
            "binning": {
                "tenure_bins": [0, 12, 24, 36, 48, 60, np.inf],
                "tenure_labels": ["0-12", "13-24", "25-36", "37-48", "49-60", "60+"]
            },
            "autopay_keywords": ["automatic", "bank transfer"]
        }
    }

def test_pipeline_initialization(sample_config):
    pipeline = DataPipeline(sample_config)
    assert pipeline.config == sample_config

def test_pipeline_initialization(sample_config):
    """
    Test that pipeline initializes correctly with config
    """
    pipeline = DataPipeline(sample_config)
    assert pipeline.config == sample_config
    
def test_pipeline_config_structure(sample_config):
    """
    Test that pipeline handles config correctly
    """
    # Test with valid config
    pipeline = DataPipeline(sample_config)
    assert "data" in pipeline.config
    assert "preprocessing" in pipeline.config
    assert isinstance(pipeline.config["data"], dict)
    assert isinstance(pipeline.config["preprocessing"], dict)

def test_data_validation(sample_config):
    """
    Test data quality and consistency checks
    """
    pipeline = DataPipeline(sample_config)
    
    # Test with invalid data
    invalid_df = pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'Churn': ['Maybe', 'No', 'Invalid'],
        'MonthlyCharges': [70.35, 'invalid', 88.40], 
        'tenure': [-1, 24, 999] 
    })
    
    with pytest.raises(Exception):
        pipeline._validate_data(invalid_df)

def _validate_data(self, df: pd.DataFrame) -> None:
    """
    Validate data quality and consistency
    """

    required_columns = ['Churn', 'MonthlyCharges', 'tenure']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate target values
    valid_target_values = {'Yes', 'No'}
    invalid_targets = set(df['Churn'].unique()) - valid_target_values
    if invalid_targets:
        raise ValueError(f"Invalid target values found: {invalid_targets}")
    
    # Validate numeric columns
    try:
        pd.to_numeric(df['MonthlyCharges'])
        pd.to_numeric(df['tenure'])
    except ValueError as e:
        raise ValueError("Invalid numeric values found in MonthlyCharges or tenure")
    
    # Validate tenure range
    tenure = pd.to_numeric(df['tenure'])
    if (tenure < 0).any() or (tenure > 100).any():
        raise ValueError("Tenure values out of expected range (0-100 months)")
