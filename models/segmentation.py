# models/segmentation.py
# ─────────────────────────────────────────────────────────────────────────────
# Customer / Product Segmentation – KMeans clustering on RFM features
#
# Input table : dw.metrics_customer_segmentation
# Output table: dw.results_segmentation
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from models.db_utils import read_table, write_results
from config import METRICS_CUSTOMER_TABLE

RFM_FEATURES = ["recency_days", "frequency", "monetary_total", "avg_units_sold"]
N_CLUSTERS   = 4   # Champions / Loyal / Potential / At-Risk


def _optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> int:
    """Finds best k using silhouette score (quick, no elbow needed)."""
    best_k, best_score = 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
        if score > best_score:
            best_score, best_k = score, k
    print(f"  Best k={best_k}  silhouette={best_score:.4f}")
    return best_k


def run_segmentation(auto_k: bool = False):
    print("\n👥 Running Customer Segmentation (KMeans)...")

    df = read_table(METRICS_CUSTOMER_TABLE)

    available = [f for f in RFM_FEATURES if f in df.columns]
    if len(available) < 2:
        print(f"  ⚠ Not enough RFM features ({available}). Aborting.")
        return

    X_raw = df[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    k = _optimal_k(X_scaled) if auto_k else N_CLUSTERS
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster_id"] = model.fit_predict(X_scaled)

    # Human-readable segment label based on monetary rank within each cluster
    cluster_means = (
        df.groupby("cluster_id")["monetary_total"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    segment_labels = {
        row["cluster_id"]: label
        for row, label in zip(
            cluster_means.to_dict("records"),
            ["Champions", "Loyal", "Potential", "At Risk"][:k],
        )
    }
    df["ml_segment"] = df["cluster_id"].map(segment_labels)

    # Cluster profile summary
    profile = df.groupby("cluster_id")[available].mean().round(2)
    profile["ml_segment"] = profile.index.map(segment_labels)
    profile["count"]      = df.groupby("cluster_id").size()
    profile = profile.reset_index()

    write_results(df,      "results_segmentation")
    write_results(profile, "results_segmentation_profile")

    print("  ✅ Segmentation done. Results saved to Azure SQL.")
    return model, df


if __name__ == "__main__":
    run_segmentation()