"""Naye transaction par prediction: python predict.py"""
import joblib, pandas as pd
import config
from data_loader import load_data

scaler = joblib.load(f"{config.MODEL_DIR}/scaler.pkl")
model = joblib.load(f"{config.MODEL_DIR}/xgboost.pkl")

df = load_data().drop(columns=[config.TARGET]).sample(5, random_state=1)
df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
df["fraud_probability"] = model.predict_proba(df)[:, 1].round(4)
df["prediction"] = (df["fraud_probability"] >= 0.5).map({True: "FRAUD", False: "Normal"})
print(df[["fraud_probability", "prediction"]])
