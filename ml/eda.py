"""Exploratory Data Analysis for Parkinson's dataset.

Generates and saves plots to `ml/plots/` and prints statistical summaries.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


RAW_CSV = Path("data/raw/parkinsons.csv")
PLOTS_DIR = Path("ml/plots")


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not RAW_CSV.exists():
        raise SystemExit(f"Raw dataset not found at {RAW_CSV}. Run data collection first.")
    return pd.read_csv(RAW_CSV)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_target_distribution(df):
    counts = df['status'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette='pastel', ax=ax)
    ax.set_title('Target class distribution (status)')
    ax.set_xlabel('status')
    ax.set_ylabel('count')
    save_fig(fig, PLOTS_DIR / 'target_distribution.png')


def plot_feature_histograms(df, numeric_cols):
    # Create histograms in small grids to avoid unreadable big figure
    cols = numeric_cols
    per_fig = 6
    for i in range(0, len(cols), per_fig):
        batch = cols[i:i+per_fig]
        n = len(batch)
        cols_grid = min(n, 3)
        rows = int(np.ceil(n/cols_grid))
        fig, axes = plt.subplots(rows, cols_grid, figsize=(4*cols_grid, 3*rows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, batch):
            sns.histplot(df[col], bins=25, kde=True, ax=ax, color='skyblue')
            ax.set_title(col)
        # hide unused axes
        for ax in axes[len(batch):]:
            ax.set_visible(False)
        idx = i // per_fig
        save_fig(fig, PLOTS_DIR / f'feature_histograms_{idx+1}.png')


def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols + ['status']].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='vlag', annot=False, fmt='.2f', ax=ax, linewidths=0.5)
    ax.set_title('Correlation matrix (including status)')
    save_fig(fig, PLOTS_DIR / 'correlation_heatmap.png')
    return corr


def plot_boxplots(df, numeric_cols, top_features=None):
    # If top_features provided, plot those; else pick 6 representative features
    if top_features is None:
        top_features = numeric_cols[:6]
    for col in top_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='status', y=col, data=df, palette='Set2', ax=ax)
        ax.set_title(f'Boxplot of {col} by status')
        save_fig(fig, PLOTS_DIR / f'boxplot_{col.replace('/', '_')}.png')


def feature_target_relationship(df, numeric_cols):
    # Plot violin/boxplots for each feature vs status (saved in small batches)
    per_fig = 6
    for i in range(0, len(numeric_cols), per_fig):
        batch = numeric_cols[i:i+per_fig]
        n = len(batch)
        cols_grid = min(n, 1)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3*n))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, batch):
            sns.violinplot(x='status', y=col, data=df, inner='quartile', palette='Set3', ax=ax)
            ax.set_title(f'{col} by status')
        idx = i // per_fig
        save_fig(fig, PLOTS_DIR / f'feature_target_relation_{idx+1}.png')


def print_statistical_analysis(df, numeric_cols):
    print("Dataset shape:", df.shape)
    print("\nDescriptive statistics (numeric features):")
    print(df[numeric_cols].describe().T)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nClass distribution:\n", df['status'].value_counts().to_dict())
    print("\nMean and std of numerical features:\n", df[numeric_cols].agg(['mean', 'std']).T)


def main():
    ensure_dirs()
    df = load_data()

    # Exclude identifier from numeric analysis
    if 'name' in df.columns:
        df_no_id = df.drop(columns=['name'])
    else:
        df_no_id = df.copy()

    # Identify numeric columns (exclude status from feature histograms when appropriate)
    numeric_cols = df_no_id.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'status' from feature lists where needed
    feature_cols = [c for c in numeric_cols if c != 'status']

    # A. Target class distribution
    plot_target_distribution(df)

    # B. Feature distributions
    plot_feature_histograms(df, feature_cols)

    # C. Correlation matrix including target
    corr = plot_correlation_heatmap(df, feature_cols)

    # D. Boxplots for important features: pick top correlated with status
    if 'status' in numeric_cols:
        corr_with_status = corr['status'].drop('status')
        top_features = corr_with_status.abs().sort_values(ascending=False).head(6).index.tolist()
    else:
        top_features = feature_cols[:6]
    plot_boxplots(df, feature_cols, top_features=top_features)

    # E. Statistical analysis (prints)
    print_statistical_analysis(df, feature_cols)

    # F. Feature-target relationship plots
    feature_target_relationship(df, feature_cols)

    print("EDA completed. Plots saved under:", PLOTS_DIR)


if __name__ == '__main__':
    main()
