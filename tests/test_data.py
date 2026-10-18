
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "heart_disease.csv"


def test_dataset_exists():
    assert DATA_FILE.exists()


def test_dataset_has_rows_and_columns():
    df = pd.read_csv(DATA_FILE)

    assert len(df) > 0
    assert len(df.columns) > 1


def test_target_is_binary():
    df = pd.read_csv(DATA_FILE)

    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_target_has_no_missing_values():
    df = pd.read_csv(DATA_FILE)

    assert df["target"].notna().all()


def test_features_are_available():
    df = pd.read_csv(DATA_FILE)
    features = df.drop(columns=["target"])

    assert len(features.columns) > 0
