
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "heart_disease.csv"
OUTPUT_DIR = ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def run_eda():
    df = pd.read_csv(DATA_FILE)

    # 1. Basic information
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe().T)

    print("\nDuplicate rows:", df.duplicated().sum())

    # 2. Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    print("\nMissing values:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    # 3. Target class balance
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="target")
    ax.set_title("Heart Disease Target Class Balance")
    ax.set_xlabel("Target (0 = absent, 1 = present)")
    ax.set_ylabel("Number of records")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "class_balance.png",
        dpi=150
    )
    plt.close()

    # 4. Histograms for the features
    features = df.drop(columns=["target"])

    axes = features.hist(
        figsize=(16, 12),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "feature_histograms.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # 5. Correlation heatmap
    plt.figure(figsize=(12, 9))
    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature and Target Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    print(f"\nEDA outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
