import pytest
import pandas as pd
import numpy as np
from data_pipeline.data_splitter import DataSplitter
from data_pipeline.imbalance_handler import ImbalanceHandler
from sklearn.preprocessing import StandardScaler

@pytest.fixture
def sample_imbalanced_data():
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
 
    y = pd.Series(np.concatenate([np.zeros(900), np.ones(100)]))
    return X, y

def test_data_splitting(sample_imbalanced_data):
    X, y = sample_imbalanced_data
    X_train, X_test, y_train, y_test = DataSplitter.split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test split sizes
    assert len(X_train) == 800
    assert len(X_test) == 200
    assert len(y_train) == 800
    assert len(y_test) == 200
    
    
    assert set(y_train.unique()) == set(y_test.unique()) == {0, 1}

def test_data_splitting_basic(sample_imbalanced_data):
    """
    Test basic data splitting functionality
    """
    X, y = sample_imbalanced_data
    test_size = 0.2
    X_train, X_test, y_train, y_test = DataSplitter.split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Test split sizes
    assert len(X_test) == int(len(X) * test_size)
    assert len(X_train) == len(X) - len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
