"""
Steps 4-7: Clustering, Characterization, Structural Analysis, Mispricing
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import binomtest
from pathlib import Path


FEATURE_COLS = [
    "final_prob", "change_3h", "change_24h", "total_range",
    "volatility", "drift_slope", "days_open", "time_near_50",
]


def run_clustering(filtered_df: pd.DataFrame, scaled_df: pd.DataFrame,
                   output_dir: str = "outputs") -> tuple[pd.DataFrame, int]:
    """Step 4: Run k-means for k=2..7, plot elbow + silhouette, return labeled df."""

    Path(output_dir).mkdir(exist_ok=True)
    print("=" * 60)
    print("STEP 4: CLUSTERING")
    print("=" * 60)

    X = scaled_df[FEATURE_COLS].values
    K_range = range(2, 8)
    inertias = []
    sil_scores = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)
        print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(K_range), inertias, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-cluster SSE)")
    ax1.set_title("Elbow Curve")
    ax1.set_xticks(list(K_range))
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_range), sil_scores, "o-", color="darkorange", linewidth=2)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.set_xticks(list(K_range))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/elbow_curve.png")

    # Select optimal k: best silhouette score among k >= 4
    # We enforce min k=4 so clusters reflect behavioral types, not just YES/NO.
    eligible_k = [(k, s) for k, s in zip(K_range, sil_scores) if k >= 4]
    best_k = max(eligible_k, key=lambda x: x[1])[0]
    best_sil = dict(zip(K_range, sil_scores))[best_k]
    print(f"\nOptimal k={best_k} (best silhouette ≥ k=4: {best_sil:.4f})")
    print("  (Global best was k=%d with %.4f; enforcing min k=4 for richer behavioral analysis)" %
          (list(K_range)[int(np.argmax(sil_scores))], max(sil_scores)))

    # Final clustering
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X)

    labeled_df = filtered_df.copy()
    labeled_df["cluster"] = cluster_labels
    labeled_df.to_csv("data/clustered_markets.csv", index=False)
    print(f"Saved: data/clustered_markets.csv ({len(labeled_df)} rows)")

    return labeled_df, best_k


def characterize_clusters(labeled_df: pd.DataFrame, best_k: int) -> pd.DataFrame:
    """Step 5: Characterize each cluster."""

    print("\n" + "=" * 60)
    print("STEP 5: CLUSTER CHARACTERIZATION")
    print("=" * 60)

    cluster_summaries = []

    for c in sorted(labeled_df["cluster"].unique()):
        sub = labeled_df[labeled_df["cluster"] == c]
        size = len(sub)
        yes_rate = (sub["result"] == "yes").mean()

        feat_means = sub[FEATURE_COLS].mean()
        dominant_type = sub["market_type"].value_counts().index[0] if "market_type" in sub.columns else "unknown"

        print(f"\n--- Cluster {c} (n={size}) ---")
        for col in FEATURE_COLS:
            print(f"  {col:20s}: {feat_means[col]:.4f}")
        print(f"  dominant market_type: {dominant_type}")
        print(f"  YES resolution rate:  {yes_rate:.3f}")

        # Behavioral description — prioritize unique features in order of distinctiveness
        fp = feat_means["final_prob"]
        vol = feat_means["volatility"]
        slope = feat_means["drift_slope"]
        t50 = feat_means["time_near_50"]
        rng = feat_means["total_range"]
        days = feat_means["days_open"]
        c24 = feat_means["change_24h"]

        if days > 7:
            desc = (f"Long-duration markets (~{days:.0f} days open). Dominant type: {dominant_type}. "
                    f"Low short-term volatility ({vol:.3f}); final prob={fp:.2f}; YES rate={yes_rate:.0%}.")
        elif fp > 0.75 and rng > 0.5:
            desc = (f"Strong late-converging favorite: started near 50/50, resolved YES ~{yes_rate:.0%} of the time "
                    f"with a large intraday range ({rng:.2f}) and decisive final-day move (+{c24:.2f} in 24h).")
        elif fp > 0.75:
            desc = (f"Stable favorite: consistently priced above 75%, resolved YES ~{yes_rate:.0%}. "
                    f"Low volatility ({vol:.3f}); limited price discovery — outcome was largely anticipated.")
        elif fp < 0.25 and t50 > 0.30:
            desc = (f"Volatile near-miss underdog: spent {t50:.0%} of lifetime near 50% before resolving NO ~{1-yes_rate:.0%} "
                    f"of the time. Large range ({rng:.2f}), strong late-day drift ({c24:.2f} in 24h). "
                    f"These markets appeared contested before the crowd shifted decisively toward NO.")
        elif fp < 0.25:
            desc = (f"Stable underdog: consistently priced below 25%, resolved NO ~{1-yes_rate:.0%} of the time. "
                    f"Low volatility ({vol:.3f}), almost no time near 50% ({t50:.0%}). "
                    f"Outcome was largely anticipated from market open.")
        elif t50 > 0.4:
            desc = (f"Contested/uncertain: spent {t50:.0%} of lifetime near 50%. Wide range ({rng:.2f}). "
                    f"Genuine price discovery with no clear directional bias. YES rate={yes_rate:.0%}.")
        elif slope > 0.01:
            desc = (f"Late-surging YES: positive drift (+{slope:.4f}/hr) in final 24h. "
                    f"Markets that moved toward YES as resolution approached. YES rate={yes_rate:.0%}.")
        elif slope < -0.01:
            desc = (f"Late-fading NO: negative drift ({slope:.4f}/hr) in final 24h. "
                    f"Markets that drifted toward NO as resolution approached. YES rate={yes_rate:.0%}.")
        else:
            desc = (f"Stable/anchored ({dominant_type}): low volatility ({vol:.3f}), limited movement. "
                    f"Final prob={fp:.2f}; YES rate={yes_rate:.0%}.")

        print(f"  Description: {desc}")

        cluster_summaries.append({
            "cluster": c,
            "size": size,
            "dominant_market_type": dominant_type,
            **{col: feat_means[col] for col in FEATURE_COLS},
            "yes_rate": yes_rate,
            "behavioral_description": desc,
        })

    return pd.DataFrame(cluster_summaries)


def structural_analysis(labeled_df: pd.DataFrame) -> None:
    """Step 6: Cross-tabulate clusters vs market type."""

    print("\n" + "=" * 60)
    print("STEP 6: STRUCTURAL ANALYSIS")
    print("=" * 60)

    if "market_type" not in labeled_df.columns:
        print("[SKIP] No market_type column found.")
        return

    ct = pd.crosstab(labeled_df["cluster"], labeled_df["market_type"], margins=True)
    print("\nCluster x Market Type cross-tabulation:")
    print(ct.to_string())

    # Normalized (row proportions)
    ct_norm = pd.crosstab(labeled_df["cluster"], labeled_df["market_type"], normalize="index").round(3)
    print("\nRow-normalized (cluster composition):")
    print(ct_norm.to_string())

    game_types = {"game_winner", "series_outcome"}
    prop_types = {"player_points", "player_prop"}

    game_markets = labeled_df[labeled_df["market_type"].isin(game_types)]
    prop_markets = labeled_df[labeled_df["market_type"].isin(prop_types)]

    if len(game_markets) > 0 and len(prop_markets) > 0:
        game_clusters = game_markets["cluster"].value_counts(normalize=True)
        prop_clusters = prop_markets["cluster"].value_counts(normalize=True)
        print("\nGame outcome market cluster distribution:")
        print(game_clusters.to_string())
        print("\nPlayer prop market cluster distribution:")
        print(prop_clusters.to_string())

        overlap = set(game_clusters.index) & set(prop_clusters.index)
        if overlap:
            print(f"\n=> Clusters {sorted(overlap)} contain BOTH game and prop markets — clusters cut ACROSS market type.")
        else:
            print("\n=> Clusters align with market type — distinct behavioral separation.")


def mispricing_analysis(labeled_df: pd.DataFrame, output_dir: str = "outputs") -> pd.DataFrame:
    """Step 7: Compute mispricing per cluster and test statistical significance."""

    print("\n" + "=" * 60)
    print("STEP 7: MISPRICING ANALYSIS")
    print("=" * 60)

    Path(output_dir).mkdir(exist_ok=True)
    rows = []

    for c in sorted(labeled_df["cluster"].unique()):
        sub = labeled_df[labeled_df["cluster"] == c]
        n = len(sub)
        avg_final_prob = sub["final_prob"].mean()
        yes_count = (sub["result"] == "yes").sum()
        yes_rate = yes_count / n
        mispricing = yes_rate - avg_final_prob

        # Binomial test: is the actual YES rate different from the predicted probability?
        bt = binomtest(int(yes_count), n, avg_final_prob, alternative="two-sided")
        p_value = bt.pvalue

        sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
        print(f"Cluster {c}: n={n}, avg_prob={avg_final_prob:.3f}, "
              f"yes_rate={yes_rate:.3f}, mispricing={mispricing:+.3f}, "
              f"p={p_value:.4f}{sig}")

        rows.append({
            "cluster":       c,
            "n":             n,
            "avg_final_prob": avg_final_prob,
            "yes_rate":      yes_rate,
            "mispricing":    mispricing,
            "p_value":       p_value,
            "significant":   p_value < 0.05,
        })

    misprice_df = pd.DataFrame(rows)
    misprice_df.to_csv(f"{output_dir}/mispricing_table.csv", index=False)
    print(f"Saved: {output_dir}/mispricing_table.csv")
    return misprice_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from features import preprocess

    features_df = pd.read_csv("data/features.csv")
    filtered_df, scaled_df, _ = preprocess(features_df)

    labeled_df, best_k = run_clustering(filtered_df, scaled_df)
    cluster_summaries = characterize_clusters(labeled_df, best_k)
    structural_analysis(labeled_df)
    misprice_df = mispricing_analysis(labeled_df)
