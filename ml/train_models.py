"""Train and compare multiple classification models on the Parkinson's dataset.

Trains: Logistic Regression, Decision Tree, Random Forest, KNN, SVM.
Saves evaluation CSV and comparison plot under `ml/` and `ml/plots/`.

This script is analysis-only and does not modify the raw dataset.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


RAW_CSV = Path("data/raw/parkinsons.csv")
PLOTS_DIR = Path("ml/plots")
OUT_CSV = Path("ml/model_comparison.csv")
OUT_PLOT = PLOTS_DIR / "model_comparison.png"


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")
    return pd.read_csv(RAW_CSV)


def build_pipeline(clf):
    # Use StandardScaler to avoid leakage by fitting only on training data inside pipeline
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    # some classifiers may not implement predict_proba; use decision_function fallback
    try:
        y_score = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_score = pipe.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc, 'confusion_matrix': cm}


def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {model_name}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


def plot_model_comparison(df_metrics, out_path=OUT_PLOT):
    # df_metrics: index=model names, columns=metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df_metrics[metrics]
    df_plot.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Model comparison')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved model comparison plot: {out_path}")


def main():
    ensure_dirs()
    df = load_data()

    if 'name' not in df.columns or 'status' not in df.columns:
        raise SystemExit("Required columns 'name' and/or 'status' not found in the dataset.")

    # Prepare features and target
    X = df.drop(columns=['name', 'status'])
    y = df['status']

    # Stratified split (same for all models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    # Define models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42, probability=True),
    }

    results = []

    for name, clf in models.items():
        print(f"\nTraining and evaluating: {name}")
        pipe = build_pipeline(clf)
        # Fit pipeline on training data (scaler fitted here)
        pipe.fit(X_train, y_train)
        res = evaluate_model(pipe, X_test, y_test)
        cm_path = PLOTS_DIR / f'confusion_{name}.png'
        plot_confusion_matrix(res['confusion_matrix'], name, cm_path)
        results.append({
            'model': name,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'roc_auc': res['roc_auc'],
        })

    results_df = pd.DataFrame(results).set_index('model')
    # Save CSV
    results_df.to_csv(OUT_CSV)
    print(f"Saved model comparison CSV: {OUT_CSV}")

    # Plot comparison
    plot_model_comparison(results_df, OUT_PLOT)

    # Print clear table
    print("\nModel evaluation results:\n")
    print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

    # Identify best model by average of metrics (simple heuristic)
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    # Replace NaN (possible for roc_auc) with -1 to avoid selection
    avg_score = results_df[metric_cols].fillna(-1).mean(axis=1)
    best_model = avg_score.idxmax()
    print(f"\nBest model (by average metric): {best_model}")


if __name__ == '__main__':
    main()
