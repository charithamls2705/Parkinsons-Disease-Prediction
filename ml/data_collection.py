"""
Download and inspect the UCI Parkinson's dataset.

Saves raw file to `data/raw/parkinsons.data` and a CSV copy
`data/raw/parkinsons.csv` for downstream steps.
"""
import os
from pathlib import Path
import requests
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
)


def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)


def download_raw(dest_path: str = "data/raw/parkinsons.data") -> None:
    ensure_dirs()
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"Saved raw data to {dest_path}")


def save_csv(raw_path: str = "data/raw/parkinsons.data", csv_path: str = "data/raw/parkinsons.csv") -> None:
    df = pd.read_csv(raw_path)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV copy to {csv_path}")


def inspect(raw_path: str = "data/raw/parkinsons.data") -> None:
    df = pd.read_csv(raw_path)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Column names:\n", "\n".join(df.columns.tolist()))
    print("\nFirst 5 rows:\n", df.head())


def main():
    dest = "data/raw/parkinsons.data"
    if not Path(dest).exists():
        download_raw(dest)
    save_csv(dest)
    inspect(dest)


if __name__ == "__main__":
    main()
