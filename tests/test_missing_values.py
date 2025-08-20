import pytest
import pandas as pd
import numpy as np
from data_pipeline.handle_missing_values import MissingValueHandler

@pytest.fixture
def sample_df_with_missing():
    return pd.DataFrame({
        'numeric_col': [1.0, np.nan, 3.0, np.nan, 5.0],
        'categorical_col': ['A', None, 'B', 'C', np.nan],
        'all_missing': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'no_missing': [1, 2, 3, 4, 5]
    })

@pytest.fixture
def mv_config():
    return {
        'strategy': 'mean',
        'numeric_to_coerce': ['numeric_col']
    }

def test_missing_value_handler_initialization(mv_config):
    mvh = MissingValueHandler(
        strategy=mv_config['strategy'],
        numeric_to_coerce=mv_config['numeric_to_coerce']
    )
    assert mvh.strategy == mv_config['strategy']
    assert mvh.numeric_to_coerce == mv_config['numeric_to_coerce']

def test_coerce_and_impute(mv_config, sample_df_with_missing):
    mvh = MissingValueHandler(
        strategy=mv_config['strategy'],
        numeric_to_coerce=mv_config['numeric_to_coerce']
    )
    imputed_df = mvh.coerce_and_impute(sample_df_with_missing.copy())
    
    
    assert imputed_df['no_missing'].equals(sample_df_with_missing['no_missing'])
    
    assert pd.to_numeric(imputed_df['numeric_col'], errors='coerce').notnull().all()
    
    
    assert all(col in imputed_df.columns for col in sample_df_with_missing.columns)
