# Credit Card Fraud Detection (Banking / FinTech)

Imbalanced data par fraud detect karne ka end-to-end ML project.
**Models:** Logistic Regression, Random Forest, XGBoost
**Imbalance handling:** SMOTE (sirf training data par)
**Metrics:** Precision, Recall, F1-score, ROC-AUC, PR-AUC

## Project Structure
| File | Kaam |
|---|---|
| `config.py` | Saari settings (paths, split, SMOTE ratio) |
| `data_loader.py` | Dataset load (nahi mila to synthetic data) |
| `preprocess.py` | Cleaning, scaling, stratified split, SMOTE |
| `models.py` | 3 models ki definition |
| `evaluate.py` | Metrics, confusion matrix, ROC/PR plots |
| `train.py` | Main pipeline - yahi run karo |
| `predict.py` | Naye transactions par prediction |

## Setup
```bash
pip install -r requirements.txt
```
Real dataset: Kaggle "Credit Card Fraud Detection" (ULB) se `creditcard.csv` download karke `data/` mein rakho.

## Run
```bash
python train.py     # train + evaluate + save models/plots
python predict.py   # sample predictions
```
Outputs: `outputs/model_comparison.csv`, `outputs/roc_pr_curves.png`, `outputs/feature_importance.png`, `models/*.pkl`

## Key Points (Viva / Interview)
1. **Accuracy misleading hai** - 99.8% normal hain, isliye Precision/Recall/F1/PR-AUC dekhte hain.
2. **SMOTE sirf train par** - test par lagane se data leakage hota hai.
3. **Scaler sirf train par fit** hota hai.
4. **Stratified split** se train/test mein same fraud ratio rehta hai.
5. **Business trade-off:** Recall high = zyada fraud pakda; Precision high = kam false alarms. Threshold isi hisaab se tune karo.
6. Note: synthetic data par scores real Kaggle data se alag aayenge (real mein ROC-AUC ~0.97+ XGBoost ka).
