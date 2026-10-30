"""Cleaning, scaling, train/test split aur SMOTE."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib, os
import config


def split_and_scale(df):
    df = df.drop_duplicates().dropna()
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]

    # stratify => train/test dono mein same fraud ratio
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    # Sirf Time & Amount scale karte hain (V1-V28 pehle se PCA-scaled)
    scaler = StandardScaler()
    cols = ["Time", "Amount"]
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[cols] = scaler.fit_transform(X_tr[cols])   # fit sirf train par (no leakage)
    X_te[cols] = scaler.transform(X_te[cols])

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{config.MODEL_DIR}/scaler.pkl")
    return X_tr, X_te, y_tr, y_te


def apply_smote(X_tr, y_tr):
    """SMOTE sirf TRAIN data par. Test data kabhi resample nahi karte."""
    sm = SMOTE(sampling_strategy=config.SMOTE_STRATEGY,
               random_state=config.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    print(f"[smote] Before: {y_tr.value_counts().to_dict()} | "
          f"After: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res
