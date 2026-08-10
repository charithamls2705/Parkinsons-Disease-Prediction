"""Inspect `data/raw/parkinsons.csv` and print requested diagnostics.

Usage: python -u ml/inspect_dataset.py
"""
from pathlib import Path
import json
import sys

FP = Path("data/raw/parkinsons.csv")

try:
    import pandas as pd
except Exception as e:
    print("MISSING_PANDAS", e)
    sys.exit(2)


def main():
    if not FP.exists():
        print(f"ERROR: file not found: {FP}")
        sys.exit(1)

    df = pd.read_csv(FP)

    out = {}
    out['shape'] = df.shape
    out['columns'] = df.columns.tolist()
    out['dtypes'] = {c: str(dt) for c, dt in df.dtypes.items()}
    out['missing_values'] = df.isnull().sum().to_dict()
    out['duplicate_rows'] = int(df.duplicated().sum())

    if 'status' in df.columns:
        out['status_unique_values'] = df['status'].unique().tolist()
        counts = df['status'].value_counts().to_dict()
        out['class_counts'] = {int(k): int(v) for k, v in counts.items()}
        total = int(df.shape[0])
        out['class_percent'] = {int(k): float(v) * 100.0 / total for k, v in counts.items()}
    else:
        out['status_unique_values'] = None

    desc = df.describe(include='number').transpose()
    out['descriptive_stats'] = desc.to_dict(orient='index')

    mins = df.min(numeric_only=True).to_dict()
    maxs = df.max(numeric_only=True).to_dict()
    out['min'] = {k: float(v) for k, v in mins.items()}
    out['max'] = {k: float(v) for k, v in maxs.items()}

    # name uniqueness
    if 'name' in df.columns:
        out['name_is_unique'] = bool(df['name'].is_unique)
    else:
        out['name_is_unique'] = None

    # detect constant or nearly constant numeric features
    const_cols = []
    near_const_cols = []
    for c in df.select_dtypes(include=['number']).columns:
        ser = df[c]
        nunique = ser.nunique(dropna=False)
        std = float(ser.std(skipna=True))
        mean = float(ser.mean(skipna=True)) if nunique > 0 else 0.0
        if nunique <= 1 or std == 0.0:
            const_cols.append(c)
        else:
            # proportion most frequent
            top_prop = ser.value_counts(normalize=True, dropna=False).iloc[0]
            if top_prop >= 0.99:
                near_const_cols.append(c)
            else:
                # relative std very small
                if abs(mean) > 0 and (std / abs(mean)) < 1e-8:
                    near_const_cols.append(c)

    out['constant_columns'] = const_cols
    out['near_constant_columns'] = near_const_cols

    out['head'] = df.head(5).to_dict(orient='records')
    out['tail'] = df.tail(5).to_dict(orient='records')

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
