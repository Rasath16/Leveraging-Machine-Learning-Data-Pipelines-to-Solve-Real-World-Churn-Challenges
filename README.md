# Advanced Telco Churn Prediction System

A production-ready, end-to-end machine learning pipeline and ensemble model suite for predicting customer churn in the telecommunications industry. Built for both transparency and scalability via Jupyter notebooks and modular Python code.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Key Features](#key-features)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Overview

This project guides users through data analysis, feature engineering, model training, evaluation, and business insight generation—all in one cohesive pipeline. It supports multiple ML algorithms, rigorous evaluation, and production-grade components for real-world deployment.

## Project Structure

├── notebooks/
│ ├── 1.1*data_assesment.ipynb
│ ├── 1.2_class_imbalance.ipynb
│ ├── 1.3_univariate.ipynb
│ ├── 1.4_bivariate*.ipynb
│ ├── 1.5_multivariate.ipynb
│ ├── 1.6_Business_insights.ipynb
│ ├── 2_data_preprocessing.ipynb
│ ├── 3_models_training.ipynb
│ └── 4_evaluation.ipynb
├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
├── data_pipeline/
├── artifacts_pipeline/
├── JoblibModels/
├── logs/
├── new_artifacts/
├── config/
│ └── config.yaml
├── tests/
│ ├── test_data_ingestion.py
│ ├── test_missing_values.py
│ ├── test_feature_engineering.py
│ └── test_pipeline.py
├── main.py
├── requirements.txt
└── README.md

## Key Features

- **Exploratory Data Analysis**: univariate, bivariate, multivariate stats and visualizations
- **Business Intelligence**: churn drivers, customer segments, revenue impact
- **Modeling Suite**: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost with hyperparameter tuning
- **Robust Evaluation**: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, feature importance
- **Production-Ready Pipeline**: modular code for ingestion, processing, imbalance handling, feature transformations
- **Testing**: comprehensive `pytest` suite for validation and maintainability

## Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Rasath16/Advanced-Telco-Churn-Prediction-System.git
   cd Advanced-Telco-Churn-Prediction-System
   ```
2. Install dependencies:
   pip install -r requirements.txt

3.Modify config parameters in config/config.yaml as needed

4.Run the automated pipeline:
python main.py

## Usage

Step through notebooks for detailed insights:

1.1_data_assesment.ipynb → data cleaning

1.2_class_imbalance.ipynb → SMOTE balancing

1.3–1.5 → EDA

1.6_Business_insights.ipynb → churn analysis

2_data_preprocessing.ipynb → feature prep

3_models_training.ipynb → model fitting

4_evaluation.ipynb → performance review

Or execute main.py for streamlined execution.

## Testing

pytest tests/ -v

Or specifically:

pytest tests/test_data_ingestion.py

Tests cover: ingestion, missing value handling, feature engineering, pipeline integrity, configuration validation.

## Contributing

Contributions are welcome! Please:

Follow existing code and folder conventions

Add corresponding tests for new features

Ensure all tests pass before submitting a PR

Update documentation when relevant
