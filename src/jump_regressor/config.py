from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "processed" / "features_round.csv",
    ROOT_DIR / "CsvFiles" / "features_round.csv",
    ROOT_DIR / "features_round.csv",
]
MODELS_DIR = ROOT_DIR / "outputs" / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

TARGET_COLUMN = "y"
START_FEATURE_INDEX = 3
RANDOM_STATE = 42
TEST_SIZE = 0.25
