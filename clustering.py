"""
Cluster patch embeddings from the SSL encoder using Leiden community detection.
Validates that the encoder learned biologically meaningful morphological structure.

Outputs:
  - UMAP plot colored by Leiden cluster
  - Heatmap of label distribution per cluster
  - Bar chart: FL1 / NotFL1 proportion per cluster
  - Spearman correlation between cluster ID and FL subtype label
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import leidenalg
import igraph as ig
from PIL import Image
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── CONFIG ───────────────────────────────────────────────────────────────────
EMBED_CSV    = "/kaggle/working/patch_embeddings.csv"   # produced by feature_extraction.py
KNN_NEIGHBORS = 15
RANDOM_STATE  = 42
# ─────────────────────────────────────────────────────────────────────────────


def build_leiden_clusters(X: np.ndarray, n_neighbors: int = KNN_NEIGHBORS) -> list:
    """Build k-NN graph and run Leiden clustering on patch embeddings."""
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    sources, targets = knn_graph.nonzero()
    g = ig.Graph(edges=list(zip(sources, targets)), directed=False)
    g.vs["name"] = list(range(X.shape[0]))
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
    cluster_ids = [0] * len(X)
    for cluster_id, nodes in enumerate(partition):
        for node in nodes:
            cluster_ids[node] = cluster_id
    return cluster_ids


def plot_umap(X: np.ndarray, cluster_ids: list):
    """UMAP scatter plot colored by Leiden cluster."""
    reducer = umap.UMAP(random_state=RANDOM_STATE)
    embedding_2d = reducer.fit_transform(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_ids, cmap="tab20", s=5)
    plt.colorbar(label="Leiden Cluster")
    plt.title("UMAP of Patch Embeddings — Leiden Clusters")
    plt.tight_layout()
    plt.show()


def plot_label_distribution(df: pd.DataFrame):
    """Heatmap and stacked bar showing FL1/NotFL1 proportion per cluster."""
    cross_tab = pd.crosstab(df["cluster_leiden"], df["label"])
    cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab_norm, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Label Distribution per Leiden Cluster")
    plt.ylabel("Leiden Cluster")
    plt.xlabel("True Label (0=NotFL1, 1=FL1)")
    plt.tight_layout()
    plt.show()

    cross_tab_norm.plot(kind="bar", stacked=True, colormap="coolwarm", figsize=(14, 6))
    plt.ylabel("Proportion of patches")
    plt.title("Label Distribution Across Leiden Clusters")
    plt.legend(["NotFL1", "FL1"], title="Label")
    plt.tight_layout()
    plt.show()


def show_cluster_samples(df: pd.DataFrame, cluster_id: int, n: int = 8):
    """Display sample patches from a given Leiden cluster."""
    paths = df[df["cluster_leiden"] == cluster_id]["path"].sample(n)
    plt.figure(figsize=(12, 3))
    for i, path in enumerate(paths):
        img = Image.open(path)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(f"Cluster {cluster_id} — sample patches")
    plt.show()


def validate_clusters(df: pd.DataFrame, X: np.ndarray):
    """
    Logistic regression on clusters to quantify separability.
    Also reports Spearman correlation between cluster ID and FL label.
    """
    y = df["cluster_leiden"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print("Cluster classification report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: Predicted vs True Clusters")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Cluster")
    plt.tight_layout()
    plt.show()

    rho, p_value = spearmanr(df["cluster_leiden"], df["label"])
    print(f"Spearman Correlation (cluster vs FL label): ρ = {rho:.4f}, p = {p_value:.4e}")


def main():
    df_embed = pd.read_csv(EMBED_CSV)
    embedding_cols = [c for c in df_embed.columns if c not in ["label", "path", "cluster_leiden"]]
    X = df_embed[embedding_cols].values

    print("Running Leiden clustering...")
    cluster_ids = build_leiden_clusters(X)
    df_embed["cluster_leiden"] = cluster_ids

    plot_umap(X, cluster_ids)
    plot_label_distribution(df_embed)
    validate_clusters(df_embed, X)

    # Inspect specific clusters manually
    show_cluster_samples(df_embed, cluster_id=5)
    show_cluster_samples(df_embed, cluster_id=7)

    # Save updated dataframe with cluster assignments
    df_embed.to_csv("/kaggle/working/patch_embeddings_clustered.csv", index=False)
    print("Saved embeddings with cluster IDs.")


if __name__ == "__main__":
    main()
