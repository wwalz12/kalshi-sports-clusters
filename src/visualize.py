"""
Step 8: Visualization
Generate all plots and save to outputs/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
from pathlib import Path


FEATURE_COLS = [
    "final_prob", "change_3h", "change_24h", "total_range",
    "volatility", "drift_slope", "days_open", "time_near_50",
]

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def cluster_scatter(labeled_df: pd.DataFrame, scaled_df: pd.DataFrame,
                    output_dir: str = "outputs") -> None:
    """PCA 2D scatter colored by cluster."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled_df[FEATURE_COLS].values)

    fig, ax = plt.subplots(figsize=(9, 6))
    clusters = sorted(labeled_df["cluster"].unique())
    for c in clusters:
        mask = labeled_df["cluster"].values == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=PALETTE[c % len(PALETTE)], label=f"Cluster {c}",
                   alpha=0.55, s=22, edgecolors="none")

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title("Kalshi NBA Markets — Behavioral Clusters (PCA)")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/cluster_scatter.png")


def feature_heatmap(labeled_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Average features per cluster as heatmap."""
    cluster_means = labeled_df.groupby("cluster")[FEATURE_COLS].mean()

    # Standardize rows for visual contrast
    normed = (cluster_means - cluster_means.mean()) / (cluster_means.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(11, max(3, len(cluster_means) * 0.9 + 1)))
    sns.heatmap(normed, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "z-score vs. global mean"})
    ax.set_title("Cluster Feature Profiles (z-scored)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/feature_heatmap.png")


def sport_breakdown(labeled_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Bar chart of market type distribution per cluster."""
    if "market_type" not in labeled_df.columns:
        print("[SKIP] No market_type column.")
        return

    ct = pd.crosstab(labeled_df["cluster"], labeled_df["market_type"], normalize="index")
    clusters = sorted(ct.index.tolist())

    fig, ax = plt.subplots(figsize=(10, max(4, len(clusters) * 1.2)))
    ct.loc[clusters].plot(kind="bar", stacked=True, ax=ax,
                          colormap="tab20", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion of Markets")
    ax.set_title("Market Type Composition by Cluster")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1), title="Market Type", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sport_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/sport_breakdown.png")


def mispricing_chart(misprice_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Mispricing per cluster with confidence intervals."""
    import matplotlib.ticker as ticker

    n_clusters = len(misprice_df)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [PALETTE[i % len(PALETTE)] for i in range(n_clusters)]
    x = np.arange(n_clusters)

    # 95% CI for proportion difference (Wald)
    n_arr = misprice_df["n"].values
    yes_rate = misprice_df["yes_rate"].values
    ci = 1.96 * np.sqrt(yes_rate * (1 - yes_rate) / (n_arr + 1e-9))

    bars = ax.bar(x, misprice_df["mispricing"].values, color=colors,
                  yerr=ci, capsize=5, edgecolor="black", linewidth=0.7, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in misprice_df["cluster"]])
    ax.set_ylabel("Mispricing (actual YES rate − avg final prob)")
    ax.set_title("Systematic Mispricing by Cluster")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.grid(True, axis="y", alpha=0.3)

    for i, row in misprice_df.iterrows():
        label = "*" if row["p_value"] < 0.05 else ""
        ax.text(i, row["mispricing"] + (ci[i] + 0.005) * np.sign(row["mispricing"] + 1e-9),
                label, ha="center", fontsize=14, color="red")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mispricing_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/mispricing_chart.png")


def calibration_curve_plot(labeled_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Calibration curve per cluster: predicted prob vs actual YES rate."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    clusters = sorted(labeled_df["cluster"].unique())
    for c in clusters:
        sub = labeled_df[labeled_df["cluster"] == c]
        if len(sub) < 20:
            continue
        y_true = (sub["result"] == "yes").astype(int).values
        y_prob = sub["final_prob"].clip(0.001, 0.999).values

        n_bins = min(8, max(3, len(sub) // 20))
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
            ax.plot(prob_pred, prob_true, "o-", color=PALETTE[c % len(PALETTE)],
                    label=f"Cluster {c} (n={len(sub)})", linewidth=2, markersize=6)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual YES Fraction")
    ax.set_title("Calibration Curves by Cluster")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/calibration_curve.png")


def generate_all(labeled_df: pd.DataFrame, scaled_df: pd.DataFrame,
                 misprice_df: pd.DataFrame, output_dir: str = "outputs") -> None:
    print("\n" + "=" * 60)
    print("STEP 8: VISUALIZATION")
    print("=" * 60)

    Path(output_dir).mkdir(exist_ok=True)

    cluster_scatter(labeled_df, scaled_df, output_dir)
    feature_heatmap(labeled_df, output_dir)
    sport_breakdown(labeled_df, output_dir)
    mispricing_chart(misprice_df, output_dir)
    calibration_curve_plot(labeled_df, output_dir)

    print("\nAll visualizations saved.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from features import preprocess

    features_df = pd.read_csv("data/features.csv")
    filtered_df, scaled_df, _ = preprocess(features_df)
    labeled_df = pd.read_csv("data/clustered_markets.csv")
    misprice_df = pd.read_csv("outputs/mispricing_table.csv")
    generate_all(labeled_df, scaled_df, misprice_df)
