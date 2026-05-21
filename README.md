# Explainova

**Explainova** is a no-code explainable machine learning platform designed for researchers who want to analyze tabular datasets without writing code.

The main goal of Explainova is not only to train machine learning models, but also to make the full analysis process visible, understandable, and reportable. The platform guides users from dataset upload to preprocessing, model comparison, explainability analysis, and Word report generation.

---

## Project Motivation

Many researchers collect valuable datasets in fields such as genetics, healthcare, education, and social sciences. However, building machine learning models usually requires coding skills, data preprocessing knowledge, metric interpretation, and explainability techniques.

Explainova aims to make this process easier by providing a guided no-code workflow for tabular machine learning analysis.

---

## Key Features

- Upload tabular datasets in CSV, XLSX, XLS, and TSV formats
- Preview dataset structure, missing values, rows, and columns
- Select target column and features manually or automatically
- Automatic preprocessing with an audit report
- Missing value handling
- Duplicate row removal
- ID-like column detection
- High-cardinality categorical feature handling
- Ordinal and one-hot encoding
- Outlier detection and capping using IQR
- Optional feature reduction using Variance Threshold and Pairwise Correlation
- Automatic classification/regression problem detection
- Multiple model comparison
- Classification metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Regression metrics: R² Score, MAE, RMSE
- Confusion matrix and ROC curve visualization
- SHAP-based explainability
- PDP/ICE feature behavior analysis
- K-Fold stability check
- Word report generation with results, charts, and explanations

---

## Why Explainova?

Explainova focuses on making machine learning accessible and explainable for non-coder researchers.

Instead of hiding the process behind a single automatic button, Explainova shows:

- how the data was cleaned,
- which features were used,
- which models were compared,
- how the metrics should be interpreted,
- which features influenced the model,
- and how the final results can be reported.

The platform is designed to support explainable and defensible machine learning analysis.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- python-docx
- openpyxl
- xlrd

---

## Supported Models

### Classification

- Logistic Regression
- SVM
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Regression

- Ridge Regression
- SVR
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Explainability Methods

Explainova uses SHAP to explain model behavior.

SHAP helps users understand:

- which features are most important,
- whether a feature pushes the prediction up or down,
- how strongly each feature affects the model output.

Explainova also includes PDP/ICE analysis to show how model predictions change when a selected feature changes.

> Note: Explainability outputs describe model behavior, not real-world causality.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/explainova.git
cd explainova
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
