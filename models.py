"""Teeno models ki definition."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import config

def get_models():
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=rs),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=rs),
    }
