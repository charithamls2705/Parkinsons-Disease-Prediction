"""Hyperparameter tuning with GridSearchCV and StratifiedKFold.

Produces `ml/tuning_results.csv` and `ml/plots/tuned_model_comparison.png`.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


RAW_CSV = Path("data/raw/parkinsons.csv")
PLOTS_DIR = Path("ml/plots")
OUT_CSV = Path("ml/tuning_results.csv")
OUT_PLOT = PLOTS_DIR / "tuned_model_comparison.png"


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")
    return pd.read_csv(RAW_CSV)


def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
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
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc}


def main():
    ensure_dirs()
    df = load_data()

    if 'name' not in df.columns or 'status' not in df.columns:
        raise SystemExit("Required columns 'name' and/or 'status' not found in dataset.")

    X = df.drop(columns=['name', 'status'])
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Train/test split:", X_train.shape, X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define parameter grids (kept moderate for runtime)
    param_grids = {
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'liblinear'],
        },
        'RandomForest': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 5, 10],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__max_features': ['sqrt', 'log2'],
        },
        'KNN': {
            'clf__n_neighbors': [3, 5, 7],
            'clf__weights': ['uniform', 'distance'],
            'clf__p': [1, 2],
        },
        'SVM': {
            'clf__C': [0.1, 1, 10],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma': ['scale', 'auto'],
        },
    }

    estimators = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42, probability=True),
    }

    results = []

    for name, clf in estimators.items():
        print(f"\nTuning {name}...")
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

        # Baseline (untuned) evaluation
        pipe.fit(X_train, y_train)
        baseline = evaluate(pipe, X_test, y_test)

        if name in param_grids:
            grid = GridSearchCV(
                pipe,
                param_grids[name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            best_params = grid.best_params_
            cv_score = grid.best_score_
            tuned_metrics = evaluate(best, X_test, y_test)
        else:
            best_params = None
            cv_score = None
            tuned_metrics = baseline

        print(f"Baseline metrics: {baseline}")
        print(f"Best params: {best_params}")
        print(f"CV best score (roc_auc): {cv_score}")
        print(f"Tuned test metrics: {tuned_metrics}")

        results.append(
            {
                'model': name,
                'baseline_accuracy': baseline['accuracy'],
                'baseline_precision': baseline['precision'],
                'baseline_recall': baseline['recall'],
                'baseline_f1': baseline['f1'],
                'baseline_roc_auc': baseline['roc_auc'],
                'best_params': json.dumps(best_params) if best_params is not None else '',
                'cv_roc_auc': float(cv_score) if cv_score is not None else np.nan,
                'tuned_accuracy': tuned_metrics['accuracy'],
                'tuned_precision': tuned_metrics['precision'],
                'tuned_recall': tuned_metrics['recall'],
                'tuned_f1': tuned_metrics['f1'],
                'tuned_roc_auc': tuned_metrics['roc_auc'],
            }
        )

    res_df = pd.DataFrame(results).set_index('model')
    res_df.to_csv(OUT_CSV)
    print(f"Saved tuning results: {OUT_CSV}")

    # Plot comparison of baseline vs tuned for key metrics
    metrics = ['baseline_accuracy', 'tuned_accuracy', 'baseline_roc_auc', 'tuned_roc_auc']
    plot_df = res_df[metrics]
    # Normalize columns for nicer plotting: show side-by-side bars
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Baseline vs Tuned: Accuracy and ROC-AUC')
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=300)
    plt.close(fig)
    print(f"Saved tuned model comparison plot: {OUT_PLOT}")

    # Print summary
    with pd.option_context('display.max_rows', None):
        print('\nTuning summary:')
        print(res_df[['best_params', 'cv_roc_auc', 'baseline_accuracy', 'tuned_accuracy', 'baseline_roc_auc', 'tuned_roc_auc']])

    # Select final candidate by comparing tuned ROC-AUC primarily, then other metrics
    # Use tuned_roc_auc where available, else baseline
    res_df['select_score'] = res_df['tuned_roc_auc'].fillna(res_df['baseline_roc_auc'])
    final_candidate = res_df['select_score'].idxmax()
    print(f"\nSelected final candidate (based on tuned ROC-AUC primarily): {final_candidate}")


if __name__ == '__main__':
    main()
