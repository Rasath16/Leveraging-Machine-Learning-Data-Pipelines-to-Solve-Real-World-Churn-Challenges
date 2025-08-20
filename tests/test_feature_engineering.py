import pytest
import pandas as pd
import numpy as np
from data_pipeline.feature_engineering import FeatureEngineering

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet'],
        'OnlineBackup': ['Yes', 'No', 'No internet'],
        'DeviceProtection': ['No', 'Yes', 'No internet'],
        'TechSupport': ['No', 'Yes', 'No internet'],
        'StreamingTV': ['Yes', 'No', 'No internet'],
        'StreamingMovies': ['Yes', 'No', 'No internet'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)']
    })

@pytest.fixture
def feature_config():
    return {
        'service_columns': [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ],
        'autopay_keywords': ['automatic', 'bank transfer']
    }

def test_feature_engineering_initialization(feature_config):
    fe = FeatureEngineering(
        service_cols=feature_config['service_columns'],
        autopay_keywords=feature_config['autopay_keywords']
    )
    assert fe.service_cols == feature_config['service_columns']
    assert fe.autopay_keywords == feature_config['autopay_keywords']

def test_add_features(feature_config, sample_df):
    fe = FeatureEngineering(
        service_cols=feature_config['service_columns'],
        autopay_keywords=feature_config['autopay_keywords']
    )
    df_with_features = fe.add_features(sample_df.copy())
    
    # Test that new features were added
    assert 'ServiceAdoptionScore' in df_with_features.columns
    assert 'IsAutoPay' in df_with_features.columns
    
    # Test that original columns were preserved
    for col in sample_df.columns:
        assert col in df_with_features.columns
