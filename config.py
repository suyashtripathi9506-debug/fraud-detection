"""Project-wide settings."""
DATA_PATH = "data/creditcard.csv"   # Kaggle dataset yahan rakho
TARGET = "Class"                    # 1 = Fraud, 0 = Normal
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_STRATEGY = 0.1                # fraud:normal ratio after SMOTE (10%)
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
