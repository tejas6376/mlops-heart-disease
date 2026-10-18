
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# Find the project root and create the data folder.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "heart_disease.csv"


def download_data():
    print("Downloading Heart Disease UCI dataset...")

    # UCI Machine Learning Repository dataset ID 45.
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()

    # The original target column is normally named "num".
    if "num" not in target.columns:
        raise ValueError(
            f"Expected target column 'num'. Found: {list(target.columns)}"
        )

    # Combine input features and target.
    df = features.copy()
    df["target"] = pd.to_numeric(
        target["num"], errors="coerce"
    ).to_numpy()

    # Convert feature columns to numeric values.
    # Unreadable values become NaN and can be handled later.
    for column in features.columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove records whose target is missing.
    df = df.dropna(subset=["target"]).copy()

    # Convert original target values 0-4 into binary labels.
    df["target"] = (df["target"] > 0).astype(int)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nTarget distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    download_data()
