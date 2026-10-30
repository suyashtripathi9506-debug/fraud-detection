# Credit Card Fraud Detection (Banking / FinTech)

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using classification algorithms. The project focuses on handling highly imbalanced datasets and evaluating fraud detection performance using appropriate metrics.

## Project Overview

Credit card fraud detection is a binary classification problem in which transactions are classified as:
- **Legitimate (0):** Normal credit card transactions.
- **Fraudulent (1):** Suspicious or unauthorized transactions.

The project uses machine learning techniques to identify fraudulent transactions while addressing class imbalance.

## Machine Learning Models

The following classification algorithms are implemented:

1. Logistic Regression
2. Random Forest
3. XGBoost

## Imbalanced Data Handling

- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training dataset.
- SMOTE is applied only to training data to prevent data leakage.
- Used stratified train-test splitting to maintain the original fraud ratio in both datasets.

## Evaluation Metrics

Model performance is evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

PR-AUC is used as an important evaluation metric because fraud detection datasets are highly imbalanced.

## Project Structure

| File | Description |
|---|---|
| `config.py` | Manages project configurations, paths, and SMOTE settings. |
| `data_loader.py` | Loads the dataset or generates synthetic data if the dataset is unavailable. |
| `preprocess.py` | Handles data cleaning, scaling, stratified splitting, and SMOTE. |
| `models.py` | Defines the machine learning classification models. |
| `evaluate.py` | Calculates evaluation metrics and generates confusion matrices and ROC/PR curves. |
| `train.py` | Executes the complete model training and evaluation pipeline. |
| `predict.py` | Predicts whether new transactions are fraudulent or legitimate. |

## Installation and Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

Download the **Credit Card Fraud Detection Dataset** from Kaggle (ULB).

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded `creditcard.csv` file inside the `data/` directory.

## How to Run

### Step 1: Train Models

```bash
python train.py
```

This command:
- Loads and preprocesses the dataset.
- Splits the data into training and testing sets.
- Applies SMOTE to the training data.
- Trains multiple machine learning models.
- Evaluates model performance.
- Saves trained models and evaluation plots.

### Step 2: Make Predictions

```bash
python predict.py
```

This command generates predictions for sample transactions using the trained models.

## Output Files

The project generates the following outputs:

- `outputs/model_comparison.csv` – Model performance comparison.
- `outputs/roc_pr_curves.png` – ROC and Precision-Recall curves.
- `outputs/feature_importance.png` – Feature importance visualization.
- `models/*.pkl` – Saved trained machine learning models.

## Key Technical Highlights

1. **Accuracy vs. Fraud Detection:** Accuracy can be misleading on highly imbalanced datasets. Precision, Recall, F1-score, and PR-AUC provide more meaningful insights.
2. **SMOTE:** Applied only to training data to prevent data leakage.
3. **Feature Scaling:** The scaler is fitted on training data only and then applied to testing data.
4. **Stratified Sampling:** Maintains a similar fraud-to-legitimate transaction ratio in training and testing datasets.
5. **Business Trade-off:** High Recall helps identify more fraudulent transactions, while high Precision reduces false fraud alerts.
6. **Threshold Optimization:** Classification thresholds can be adjusted according to business requirements.

**Note:** Performance scores obtained using synthetic data may differ significantly from results on the original Kaggle dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Joblib
- Git & GitHub

## Project Objective

To build a reliable machine learning pipeline that identifies potentially fraudulent credit card transactions, handles imbalanced data, and supports data-driven fraud detection in the Banking and FinTech domain.
