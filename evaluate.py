"""Metrics + plots: Precision, Recall, F1, ROC-AUC, PR-AUC."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             classification_report)
import config


def evaluate_model(name, model, X_te, y_te, threshold=0.5):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    res = {
        "Model": name,
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred),
        "F1-score": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "PR-AUC": average_precision_score(y_te, proba),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_te, pred, target_names=["Normal", "Fraud"], digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    return res, proba


def save_plots(probas, y_te):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    for a in ax: a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{config.OUTPUT_DIR}/roc_pr_curves.png", dpi=130); plt.close()


def save_results(rows):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(f"{config.OUTPUT_DIR}/model_comparison.csv", index=False)
    print("\n=== FINAL COMPARISON ===\n", df.to_string(index=False))
    return df


def save_feature_importance(model, cols):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=cols).nlargest(15)[::-1]
    imp.plot(kind="barh", figsize=(7, 6), title="XGBoost - Top 15 Features")
    plt.tight_layout(); plt.savefig(f"{config.OUTPUT_DIR}/feature_importance.png", dpi=130); plt.close()
