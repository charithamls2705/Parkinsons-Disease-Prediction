"""Feature analysis for Parkinson's dataset.

Generates:
- Correlation of each numeric feature with `status` (sorted by absolute value)
- Statistical comparison (mean, std, median) per class for each feature
- Preliminary feature importances via RandomForest (no tuning)
- Saves plot `ml/plots/feature_importance.png` and CSV `ml/feature_importance.csv`

This module is analysis-only and does not modify the raw dataset.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RAW_CSV = Path("data/raw/parkinsons.csv")
PLOTS_DIR = Path("ml/plots")
OUT_CSV = Path("ml/feature_importance.csv")
OUT_PLOT = PLOTS_DIR / "feature_importance.png"


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")
    return pd.read_csv(RAW_CSV)


def correlation_with_target(df, feature_cols):
    # include status
    corr = df[feature_cols + ['status']].corr()
    corr_with_status = corr['status'].drop('status')
    corr_df = corr_with_status.rename('correlation').to_frame()
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_corr', ascending=False)
    return corr_df


def statistical_comparison(df, feature_cols):
    stats = []
    for col in feature_cols:
        g0 = df[df['status'] == 0][col]
        g1 = df[df['status'] == 1][col]
        stats.append(
            {
                'feature': col,
                'mean_0': g0.mean(),
                'std_0': g0.std(),
                'median_0': g0.median(),
                'mean_1': g1.mean(),
                'std_1': g1.std(),
                'median_1': g1.median(),
                'diff_means': g1.mean() - g0.mean(),
            }
        )
    return pd.DataFrame(stats).set_index('feature')


def compute_feature_importance(X, y, feature_cols):
    # stratified split for importance estimation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train[feature_cols], y_train)
    importances = rf.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    return fi


def plot_feature_importance(fi_series, out_path=OUT_PLOT, top_n=20):
    top = fi_series.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3*len(top))))
    sns.barplot(x=top.values, y=top.index, palette='viridis', ax=ax)
    ax.set_title('Feature importance (Random Forest)')
    ax.set_xlabel('importance')
    ax.set_ylabel('feature')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved feature importance plot: {out_path}")


def main():
    ensure_dirs()
    df = load_data()

    if 'name' not in df.columns or 'status' not in df.columns:
        raise SystemExit("Required columns 'name' and/or 'status' not found.")

    # Exclude name identifier
    df_features = df.drop(columns=['name'])

    # Numeric input features (exclude status)
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != 'status']

    print("Dataset shape:", df.shape)
    print("Number of numeric features:", len(feature_cols))

    # A. Correlation analysis
    corr_df = correlation_with_target(df, feature_cols)
    print("\nCorrelation of features with status (sorted by absolute correlation):\n")
    print(corr_df[['correlation']])

    # B. Statistical comparison between classes
    stats_df = statistical_comparison(df, feature_cols)
    print("\nStatistical comparison (mean/std/median) by status:\n")
    with pd.option_context('display.max_rows', None, 'display.float_format', '{:.6f}'.format):
        print(stats_df)

    # C. Feature importance via Random Forest (preliminary)
    X = df[feature_cols]
    y = df['status']
    fi = compute_feature_importance(X, y, feature_cols)
    fi_df = fi.rename('importance').to_frame()
    fi_df.to_csv(OUT_CSV)
    print(f"Saved feature importance CSV: {OUT_CSV}")

    # D. Visualization
    plot_feature_importance(fi, OUT_PLOT)

    # E. Save results already done (CSV)

    # Summary explanation: print top features and short reasoning
    top_feats = fi.head(10)
    print("\nTop features by importance:\n", top_feats)
    print("\nBrief explanation: features listed above have the highest importance according to the Random Forest."
          " High importance means the feature contributed more to decreasing impurity across trees."
          " Features that also show strong correlation with `status` and notable differences in class-wise means"
          " are likely informative. Use these results to guide feature selection and further analysis, but do not"
          " treat this preliminary RF as a final production model.")


if __name__ == '__main__':
    main()
