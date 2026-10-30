"""Data load karta hai. Agar Kaggle file nahi mili to synthetic data banata hai."""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import config


def _make_synthetic(n=60000, fraud_rate=0.0035):
    """creditcard.csv jaisa structure: Time, V1-V28, Amount, Class."""
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=12, n_redundant=6,
        weights=[1 - fraud_rate, fraud_rate], class_sep=1.2,
        flip_y=0.001, random_state=config.RANDOM_STATE)
    rng = np.random.default_rng(config.RANDOM_STATE)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(rng.integers(0, 172800, n))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.3, n), 2)
    df[config.TARGET] = y
    return df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", config.TARGET]]


def load_data():
    if os.path.exists(config.DATA_PATH):
        print(f"[data] Real dataset loaded: {config.DATA_PATH}")
        return pd.read_csv(config.DATA_PATH)
    print("[data] creditcard.csv nahi mili -> synthetic data use ho raha hai "
          "(real results ke liye Kaggle dataset data/ folder mein rakho)")
    return _make_synthetic()


def summarize(df):
    vc = df[config.TARGET].value_counts()
    print(f"[data] Shape: {df.shape} | Normal: {vc.get(0,0)} | Fraud: {vc.get(1,0)} "
          f"({vc.get(1,0)/len(df)*100:.3f}%)")
