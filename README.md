# Advanced Telco Churn Prediction System

A comprehensive end-to-end machine learning system for predicting customer churn in the telecommunications industry, primarily implemented through interactive Jupyter notebooks for transparency and reproducibility.

## Notebook Implementation

The core ML system is implemented through a series of well-structured Jupyter notebooks that provide a complete workflow from data analysis to model deployment:

### 1. Data Analysis and Preprocessing

- **Data Assessment** (`1.1_data_assesment.ipynb`)

  - Initial data loading and quality assessment
  - Missing value analysis and handling
  - Basic data cleaning and preprocessing

- **Class Imbalance Analysis** (`1.2_class_imbalance.ipynb`)
  - Analysis of target variable distribution
  - Implementation of SMOTE for handling imbalanced data
  - Validation of balanced dataset

### 2. Exploratory Data Analysis (EDA)

- **Univariate Analysis** (`1.3_univariate.ipynb`)
  - Distribution analysis of individual features
  - Statistical summaries with visualizations
  - Feature quality assessment
- **Bivariate Analysis** (`1.4_bivariate_.ipynb`)
  - Feature correlation analysis
  - Target variable relationships
  - Statistical dependency tests
  - Pair-wise feature interactions
- **Multivariate Analysis** (`1.5_multivariate.ipynb`)
  - Complex feature interactions
  - Dimensionality analysis
  - Feature importance preliminary assessment
  - Multi-feature correlation patterns

### 3. Business Intelligence

- **Business Insights** (`1.6_Business_insights.ipynb`)
  - Customer behavior patterns
  - Churn risk factors identification
  - Revenue impact analysis
  - Actionable business recommendations
  - Customer segmentation insights

### 4. Model Development and Evaluation

- **Data Preprocessing** (`2_data_preprocessing.ipynb`)
  - Feature engineering pipeline
  - Feature encoding and scaling
  - Train-test split implementation
  - Data validation checks
- **Model Training** (`3_models_training.ipynb`)
  - Multiple model implementations:
    - Logistic Regression (baseline)
    - Decision Trees with optimizations
    - Random Forest (basic and tuned)
    - XGBoost (basic and tuned)
    - CatBoost (basic and tuned)
  - Comprehensive hyperparameter tuning
  - Cross-validation implementation
  - Feature importance analysis
- **Model Evaluation** (`4_evaluation.ipynb`)
  - Comprehensive model comparison
  - Performance metrics analysis:
    - Accuracy, Precision, Recall
    - F1-score, ROC-AUC
    - Confusion matrices
  - Feature importance visualization
  - Model interpretation and insights
  - Business impact assessment

## Project Structure

```
├── notebooks/            # Core ML implementation notebooks
├── data/                # Data directory
│   ├── raw/            # Original dataset
│   ├── processed/      # Cleaned datasets
│   └── external/       # External data sources
├── artifacts_pipeline/  # Model artifacts
├── JoblibModels/       # Trained models
├── data_pipeline/      # Supporting pipeline code
├── config/             # Configuration files
├── logs/              # Application logs
└── new_artifacts/     # Additional model artifacts
```

## Implementation Details

### Model Training Pipeline

The model training pipeline (`3_models_training.ipynb`) implements:

1. **Advanced Model Training**

   - Multiple model architectures
   - Comprehensive hyperparameter tuning
   - Cross-validation for robust evaluation
   - Feature importance analysis

2. **Model Evaluation Metrics**

   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC analysis
   - Confusion matrix visualization
   - Cross-validation scores

3. **Feature Importance Analysis**
   - Model-specific importance scores
   - Visual importance rankings
   - Feature impact interpretation

### 1. Data Analysis and Preprocessing

- **Data Assessment** (`1.1_data_assesment.ipynb`)
  - Initial data loading and quality assessment
  - Missing value analysis
  - Basic data cleaning and preprocessing
- **Class Imbalance Analysis** (`1.2_class_imbalance.ipynb`)
  - Analysis of target variable distribution
  - Implementation of SMOTE for handling imbalanced data
  - Validation of balanced dataset

### 2. Exploratory Data Analysis (EDA)

- **Univariate Analysis** (`1.3_univariate.ipynb`)
  - Distribution analysis of individual features
  - Statistical summaries and visualizations
- **Bivariate Analysis** (`1.4_bivariate_.ipynb`)
  - Feature correlation analysis
  - Target variable relationships
  - Statistical dependency tests
- **Multivariate Analysis** (`1.5_multivariate.ipynb`)
  - Complex feature interactions
  - Dimensionality analysis
  - Feature importance preliminary assessment

### 3. Business Intelligence

- **Business Insights** (`1.6_Business_insights.ipynb`)
  - Customer behavior patterns
  - Churn risk factors
  - Revenue impact analysis
  - Actionable business recommendations

### 4. Model Development

- **Data Preprocessing** (`2_data_preprocessing.ipynb`)
  - Feature engineering pipeline
  - Feature encoding and scaling
  - Train-test split implementation
- **Model Training** (`3_models_training.ipynb`)
  - Implementation of multiple models:
    - Logistic Regression (baseline)
    - Decision Trees
    - Random Forest (basic and tuned)
    - XGBoost (basic and tuned)
    - CatBoost (basic and tuned)
  - Hyperparameter tuning
  - Cross-validation implementation
- **Model Evaluation** (`4_evaluation.ipynb`)
  - Comprehensive model comparison
  - Performance metrics analysis
  - Feature importance visualization
  - Model interpretation and insights

## Models Implemented

The project includes various machine learning models:

- CatBoost (Basic and Tuned versions)
- Random Forest (Basic and Tuned versions)
- XGBoost (Basic and Tuned versions)
- Decision Tree
- Logistic Regression

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- PyYAML
- Joblib
- Imbalanced-learn
- SciPy
- Pytest (for testing)

## Setup and Installation

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the pipeline parameters in `config/config.yaml`
4. Run the main pipeline:
   ```bash
   python main.py
   ```

## Supporting Components

### Production Data Pipeline

The notebook implementation is supported by a modular data pipeline (`data_pipeline/`) for production deployment:

1. **Data Processing Components**

   - Data Ingestion (`data_ingestion.py`)
     - Automated data loading
     - Basic cleaning operations
   - Missing Value Handler (`handle_missing_values.py`)
     - Multiple imputation strategies
     - Data validation

2. **Feature Processing**

   - Feature Engineering (`feature_engineering.py`)
     - Automated feature creation
     - Domain-specific transformations
   - Feature Binning (`feature_binning.py`)
     - Continuous variable discretization
     - Custom binning strategies
   - Feature Encoding (`feature_encoding.py`)
     - Categorical encoding
     - Scaling operations

3. **Production Support**
   - Imbalance Handler (`imbalance_handler.py`)
     - Automated SMOTE implementation
     - Class balance management
   - Pipeline Automation
     - Scalable data processing
     - Reproducible workflows

### Model Artifacts

- Trained models stored in `JoblibModels/`
- Preprocessing objects in `artifacts_pipeline/`
- Evaluation results and metrics
- Feature importance data

## Testing Framework

The project includes a comprehensive testing suite that ensures reliability and correctness of the data pipeline components. The tests are implemented using pytest and focus on key functionality verification.

### Test Structure

```
tests/
├── test_data_ingestion.py      # Tests for data loading and cleaning
├── test_feature_engineering.py  # Tests for feature creation
├── test_missing_values.py      # Tests for missing value handling
├── test_data_processing.py     # Tests for data splitting
├── test_pipeline.py           # Tests for pipeline configuration
└── __init__.py                # Test package initialization
```

### Key Test Components

1. **Data Ingestion Tests**

   - Configuration validation
   - Data loading verification
   - Basic cleaning operations
   - Column validation

2. **Feature Engineering Tests**

   - Feature creation verification
   - Input/output validation
   - Column preservation checks
   - Feature naming validation

3. **Missing Value Tests**

   - Imputation strategy verification
   - Data type preservation
   - Column consistency checks
   - Non-missing value preservation

4. **Data Processing Tests**

   - Train-test splitting validation
   - Data size verification
   - Random state consistency
   - Split ratio validation

5. **Pipeline Tests**
   - Configuration structure validation
   - Pipeline initialization
   - Basic pipeline functionality
   - Error handling

### Running Tests

Execute the full test suite:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_data_ingestion.py
pytest tests/test_pipeline.py
```

### Test Design Principles

- **Simplicity**: Each test focuses on one specific functionality
- **Reliability**: Tests avoid brittle assertions and complex setup
- **Maintainability**: Clear test names and documentation
- **Independence**: Tests can run in isolation
- **Coverage**: Core functionality is well-tested

Tests are designed to be:

- Easy to understand and maintain
- Quick to execute
- Reliable across different environments
- Focused on essential functionality

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

- Follow the existing code structure
- Add appropriate tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting
