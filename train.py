"""Main entry point:  python train.py"""
import joblib, os
import config
from data_loader import load_data, summarize
from preprocess import split_and_scale, apply_smote
from models import get_models
from evaluate import evaluate_model, save_plots, save_results, save_feature_importance


def main():
    df = load_data()
    summarize(df)
    X_tr, X_te, y_tr, y_te = split_and_scale(df)
    X_res, y_res = apply_smote(X_tr, y_tr)

    rows, probas, trained = [], {}, {}
    for name, model in get_models().items():
        print(f"\n[train] {name} ...")
        model.fit(X_res, y_res)
        res, p = evaluate_model(name, model, X_te, y_te)
        rows.append(res); probas[name] = p; trained[name] = model
        joblib.dump(model, f"{config.MODEL_DIR}/{name.replace(' ', '_').lower()}.pkl")

    table = save_results(rows)
    save_plots(probas, y_te)
    save_feature_importance(trained["XGBoost"], X_tr.columns)
    best = table.sort_values("PR-AUC", ascending=False).iloc[0]["Model"]
    print(f"\n[done] Best model (PR-AUC): {best}")
    print(f"[done] Outputs -> {config.OUTPUT_DIR}/ | Models -> {config.MODEL_DIR}/")


if __name__ == "__main__":
    main()
