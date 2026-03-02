"""
eda.py — Task 1: Exploratory Data Analysis.

Produces:
  • Class balance bar chart
  • Per-feature distribution histograms (signal vs background overlay)
  • Correlation heatmap
  • Summary statistics table
All outputs saved to ``outputs/eda/``.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import click

from lsda.config import (
    FEATURE_COLS, LOW_LEVEL_FEATURES, HIGH_LEVEL_FEATURES,
    LABEL_COL, EDA_DIR, ensure_dirs,
)
from lsda.data import load_pandas


def run_eda() -> None:
    """Execute the full EDA pipeline and save artefacts."""
    ensure_dirs()
    df = load_pandas("train")

    click.echo("\n── 1. Summary statistics ──")
    _summary_statistics(df)

    click.echo("\n── 2. Class balance ──")
    _class_balance(df)

    click.echo("\n── 3. Feature distributions ──")
    _feature_distributions(df)

    click.echo("\n── 4. Correlation heatmap ──")
    _correlation_heatmap(df)

    click.echo(f"\n✔ EDA complete — outputs saved to {EDA_DIR}")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _summary_statistics(df: pd.DataFrame) -> None:
    """Compute and save descriptive statistics for every feature."""
    stats = df[FEATURE_COLS].describe().T
    stats["missing_%"] = df[FEATURE_COLS].isnull().mean() * 100
    stats_path = EDA_DIR / "summary_statistics.csv"
    stats.to_csv(stats_path)
    click.echo(f"  Saved → {stats_path}")

    # Also print a compact view
    click.echo(stats.to_string())


def _class_balance(df: pd.DataFrame) -> None:
    """Bar chart showing signal vs background counts."""
    counts = df[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=ax, color=["#3498db", "#e74c3c"], edgecolor="black")
    ax.set_xticklabels(["Background (0)", "Signal (1)"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Class Balance — Signal vs Background")
    for i, v in enumerate(counts):
        ax.text(i, v + 5000, f"{v:,}", ha="center", fontsize=10)
    fig.tight_layout()
    path = EDA_DIR / "class_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    click.echo(f"  Saved → {path}")


def _feature_distributions(df: pd.DataFrame) -> None:
    """Overlay histograms of each feature split by label."""
    # Low-level features
    _plot_feature_group(df, LOW_LEVEL_FEATURES, "low_level_distributions.png",
                        "Low-Level Kinematic Feature Distributions")
    # High-level features
    _plot_feature_group(df, HIGH_LEVEL_FEATURES, "high_level_distributions.png",
                        "High-Level Physics-Derived Feature Distributions")


def _plot_feature_group(df: pd.DataFrame, features: list[str],
                        filename: str, suptitle: str) -> None:
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    sig = df[df[LABEL_COL] == 1]
    bkg = df[df[LABEL_COL] == 0]

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(bkg[feat], bins=50, alpha=0.5, label="Background", color="#3498db", density=True)
        ax.hist(sig[feat], bins=50, alpha=0.5, label="Signal", color="#e74c3c", density=True)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    path = EDA_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"  Saved → {path}")


def _correlation_heatmap(df: pd.DataFrame) -> None:
    """Full correlation matrix heatmap."""
    corr = df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = EDA_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    click.echo(f"  Saved → {path}")
