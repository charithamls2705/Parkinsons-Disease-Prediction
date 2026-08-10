"""Preprocessing for Parkinson's dataset.

This script loads `data/raw/parkinsons.csv`, separates features and target,
splits into train/test, fits a `StandardScaler` on training features and
transforms both splits. The fitted scaler is saved to `models/scaler.joblib`.

Notes:
- Does NOT modify the raw CSV.
- Keeps `name` as identifier (excluded from features).
- Uses `status` as the target.
- Saves processed datasets under `data/processed/` for later experiments.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_CSV = Path("data/raw/parkinsons.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load CSV
    df = pd.read_csv(RAW_CSV)

    print("Original dataset shape:", df.shape)

    # 2. Identify identifier and target
    if 'name' not in df.columns or 'status' not in df.columns:
        raise SystemExit("Required columns 'name' and/or 'status' not found in the dataset.")

    # 3. Keep name as identifier (do not use it as feature)
    identifier = df['name'].copy()

    # 4. Separate X and y
    X = df.drop(columns=['name', 'status'])
    y = df['status'].copy()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 5. Check missing values and duplicates
    missing_per_column = df.isnull().sum()
    total_missing = int(missing_per_column.sum())
    duplicate_rows = int(df.duplicated().sum())
    print("Total missing values in dataset:", total_missing)
    print("Missing values per column:\n", missing_per_column.to_dict())
    print("Duplicate rows:", duplicate_rows)

    # 6. Train/test split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        identifier,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training set shape (X_train, y_train):", X_train.shape, y_train.shape)
    print("Testing set shape  (X_test, y_test):", X_test.shape, y_test.shape)

    # Class distribution
    class_counts = y.value_counts().to_dict()
    print("Class distribution (full dataset):", class_counts)

    # 7. Scaling: fit scaler only on training numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    # Fit on training data numeric columns
    scaler.fit(X_train[numeric_cols])

    # Transform both train and test numeric columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 8. Save scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")

    # 9. Save processed datasets for later experiments (do not overwrite raw)
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    # Save identifiers too (useful to map back predictions)
    id_train.to_frame(name='name').to_csv(PROCESSED_DIR / "id_train.csv", index=False)
    id_test.to_frame(name='name').to_csv(PROCESSED_DIR / "id_test.csv", index=False)

    print("Saved processed train/test CSVs under:", PROCESSED_DIR)

    print("Preprocessing completed successfully.")


if __name__ == '__main__':
    main()
