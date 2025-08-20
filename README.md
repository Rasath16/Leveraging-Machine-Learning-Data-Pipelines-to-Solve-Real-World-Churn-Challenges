# 📊 Advanced Telco Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Sklearn%2C%20XGBoost%2C%20CatBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A **production-ready, end-to-end machine learning pipeline** for predicting customer churn in the telecommunications industry.  
This project combines **exploratory data analysis, feature engineering, advanced ML models, evaluation metrics, and business insights** into a single streamlined system.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

---

## 🔎 Overview

Customer churn is a major challenge in the telecom industry, directly impacting profitability.  
This project predicts churners using advanced ML models and delivers **actionable business insights** to help companies design retention strategies.

It includes:

- Modular Python pipeline for **production deployment**
- Jupyter notebooks for **EDA, preprocessing, training, and evaluation**
- Business intelligence to **quantify churn impact**

---

## 📂 Project Structure

```

├── notebooks/
│   ├── 1.1\_data\_assesment.ipynb
│   ├── 1.2\_class\_imbalance.ipynb
│   ├── 1.3\_univariate.ipynb
│   ├── 1.4\_bivariate\_.ipynb
│   ├── 1.5\_multivariate.ipynb
│   ├── 1.6\_Business\_insights.ipynb
│   ├── 2\_data\_preprocessing.ipynb
│   ├── 3\_models\_training.ipynb
│   └── 4\_evaluation.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── data\_pipeline/          # Production data processing modules
├── artifacts\_pipeline/     # Preprocessing artifacts
├── JoblibModels/           # Saved models
├── logs/                   # Pipeline logs
├── new\_artifacts/          # Additional model outputs
├── config/
│   └── config.yaml         # Configuration file
├── tests/                  # Unit tests (pytest)
├── main.py                 # Run the full pipeline
├── requirements.txt        # Dependencies
└── README.md

```

---

## ⚡ Features

✅ **Exploratory Data Analysis (EDA):** Univariate, bivariate, and multivariate visualizations  
✅ **Imbalanced Data Handling:** SMOTE oversampling for churn class balance  
✅ **Business Intelligence:** Identify churn drivers, revenue impact, and customer risk segments  
✅ **Modeling Suite:** Logistic Regression, Decision Trees, Random Forests, XGBoost, CatBoost  
✅ **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix  
✅ **Feature Importance:** Model interpretability to explain churn predictions  
✅ **Production Ready:** Modular pipeline with config management & logging  
✅ **Testing:** Automated unit tests with `pytest`

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
# Clone repository
git clone https://github.com/Rasath16/Advanced-Telco-Churn-Prediction-System.git
cd Advanced-Telco-Churn-Prediction-System

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Option 1: Run the full automated pipeline

```bash
python main.py
```

### Option 2: Step through Jupyter notebooks

1. `1.1_data_assesment.ipynb` → Assess data quality
2. `1.2_class_imbalance.ipynb` → Handle imbalance with SMOTE
3. `1.3_univariate.ipynb` → Univariate EDA
4. `1.4_bivariate_.ipynb` → Bivariate EDA
5. `1.5_multivariate.ipynb` → Multivariate EDA
6. `1.6_Business_insights.ipynb` → Generate churn insights
7. `2_data_preprocessing.ipynb` → Preprocessing & feature engineering
8. `3_models_training.ipynb` → Model training & tuning
9. `4_evaluation.ipynb` → Model evaluation & comparison

---

## 🧪 Testing

Run the unit tests with **pytest**:

```bash
pytest tests/ -v
```

Example (test data ingestion only):

```bash
pytest tests/test_data_ingestion.py
```

Tests cover:

- Data ingestion validation
- Missing value handling
- Feature engineering
- Pipeline consistency

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

Please ensure:

- Tests are added/updated
- Code style is consistent
- Documentation is updated

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

👤 **Author:** [Rasath16](https://github.com/Rasath16)

If you have questions or suggestions, feel free to open an **issue** or reach out via GitHub. 🚀
