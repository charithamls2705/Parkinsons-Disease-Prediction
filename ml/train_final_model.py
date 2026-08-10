"""Train and save the final production KNN pipeline for Parkinson's prediction.

This script reads best KNN hyperparameters from `ml/tuning_results.csv`,
builds a pipeline with standard scaling, trains it on the complete dataset,
and saves both the trained pipeline and the ordered feature names.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


RAW_CSV = Path("data/raw/parkinsons.csv")
TUNING_CSV = Path("ml/tuning_results.csv")
MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "parkinsons_knn_pipeline.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"


def load_knn_params():
    if not TUNING_CSV.exists():
        raise SystemExit(f"Tuning results not found at {TUNING_CSV}. Run ml/tune_models.py first.")
    tuning_df = pd.read_csv(TUNING_CSV)
    knn_row = tuning_df[tuning_df['model'] == 'KNN']
    if knn_row.empty:
        raise SystemExit("Could not find KNN tuning results in ml/tuning_results.csv")
    params_json = knn_row.iloc[0]['best_params']
    params = json.loads(params_json)
    # Remove pipeline prefixes like 'clf__'
    knn_params = {k.split('clf__')[-1]: v for k, v in params.items()}
    return knn_params


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV)
    if 'name' not in df.columns or 'status' not in df.columns:
        raise SystemExit("Expected columns 'name' and 'status' in the dataset.")

    X = df.drop(columns=['name', 'status'])
    y = df['status']

    knn_params = load_knn_params()

    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**knn_params, n_jobs=-1)),
        ]
    )

    pipeline.fit(X, y)

    joblib.dump(pipeline, PIPELINE_PATH)

    feature_names = X.columns.tolist()
    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2)

    print(f"Selected model: KNeighborsClassifier")
    print(f"Selected KNN parameters: {knn_params}")
    print(f"Number of input features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Saved pipeline to: {PIPELINE_PATH}")
    print("Training completed successfully.")

    # Verification step
    loaded_pipeline = joblib.load(PIPELINE_PATH)
    sample = X.iloc[[0]]
    prediction = loaded_pipeline.predict(sample)
    print(f"Verification prediction for first sample: {prediction[0]}")
    print("Final pipeline verification succeeded.")


if __name__ == '__main__':
    main()
