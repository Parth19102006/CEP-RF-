import pandas as pd
from pathlib import Path

DATASET_DIR = Path("archive")

files = sorted(DATASET_DIR.glob("*.parquet"))

print("=" * 70)
print("CIC-DDoS2019 DATASET INSPECTION")
print("=" * 70)

print(f"\nParquet files found: {len(files)}")

for file in files:
    print(f"  - {file.name}")

print("\n" + "=" * 70)
print("FILE INFORMATION")
print("=" * 70)

for file in files:
    df = pd.read_parquet(file)

    print(f"\n{file.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    if "Label" in df.columns:
        print("  Labels:")
        print(df["Label"].value_counts().to_string())

print("\n" + "=" * 70)
print("CHECKING SYN TRAINING")
print("=" * 70)

df = pd.read_parquet(DATASET_DIR / "Syn-training.parquet")

print(f"\nShape: {df.shape}")

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nUnique labels:")
print(df["Label"].unique())

print("\nInfinite values:")
numeric_df = df.select_dtypes(include="number")

print(numeric_df.isin([float("inf"), float("-inf")]).sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())