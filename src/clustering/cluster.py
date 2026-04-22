# kmeans etc

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(__file__).resolve().parents[2] / "results"


def reduce_dimensions(X, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_red = svd.fit_transform(X)
    ev = svd.explained_variance_ratio_.sum()
    print(f"  SVD to {n_components} dims, explained variance: {ev:.3f}")
    X_red = StandardScaler().fit_transform(X_red)
    return X_red


def run_kmeans(X, y_true, k_range=range(5, 35, 5)):
    results = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        nmi    = normalized_mutual_info_score(y_true, labels)
        ari    = adjusted_rand_score(y_true, labels)
        print(f"    K={k}: silhouette={sil:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
        results.append({"k": k, "silhouette": sil, "nmi": nmi, "ari": ari})
    return results


def run_hierarchical(X, y_true, n_clusters=24):
    """ward linkage, fixed k=24 to match category count"""
    agg    = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    lbl    = agg.fit_predict(X)
    sil    = silhouette_score(X, lbl, sample_size=min(2000, len(X)))
    nmi    = normalized_mutual_info_score(y_true, lbl)
    ari    = adjusted_rand_score(y_true, lbl)
    print(f"  Hierarchical (k={n_clusters}): silhouette={sil:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
    return {"method": "Hierarchical", "k": n_clusters, "silhouette": sil, "nmi": nmi, "ari": ari}


def run_dbscan(X, y_true, eps_range=(3.0, 4.0, 5.0, 6.0, 8.0)):
    res = []
    for eps in eps_range:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        if n_clusters >= 2:
            mask = labels != -1
            sil = silhouette_score(X[mask], labels[mask], sample_size=min(2000, mask.sum()))
            nmi = normalized_mutual_info_score(y_true, labels)
        else:
            sil = -1
            nmi = 0
        ari = adjusted_rand_score(y_true, labels)
        res.append({"eps": eps, "n_clusters": n_clusters, "n_noise": n_noise,
                     "silhouette": sil, "nmi": nmi, "ari": ari})
        print(f"    eps={eps}: clusters={n_clusters}, noise={n_noise}, sil={sil:.4f}, NMI={nmi:.4f}")
    return res


def plot_tsne(X, labels, color_name, title, filename):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab20", s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved t-SNE plot to {path}")


def run_all_clustering(X_flat, X_section, y_true, label_encoder):
    all_results = []

    for mode, X_sparse in [("flat", X_flat), ("section", X_section)]:
        print(f"\n--- Clustering: {mode} ---")
        X_dense = reduce_dimensions(X_sparse, n_components=50)

        print(f"  K-Means sweep:")
        km_res = run_kmeans(X_dense, y_true)
        for r in km_res:
            all_results.append({"mode": mode, "method": "KMeans", **r})

        hier = run_hierarchical(X_dense, y_true, n_clusters=24)
        all_results.append({"mode": mode, **hier})

        print(f"  DBSCAN sweep:")
        db_res = run_dbscan(X_dense, y_true)
        for r in db_res:
            all_results.append({"mode": mode, "method": "DBSCAN", **r})

        best_k  = max(km_res, key=lambda r: r["silhouette"])["k"]
        km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_dense)
        print(f"  best k by silhouette: {best_k}")

        plot_tsne(X_dense, km_best.labels_, "cluster",
                  f"t-SNE: {mode} K-Means (k={best_k})",
                  f"tsne_clusters_{mode}.png")
        plot_tsne(X_dense, y_true, "category",
                  f"t-SNE: {mode} True Categories",
                  f"tsne_true_{mode}.png")

    results_df = pd.DataFrame(all_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / "clustering_comparison.csv", index=False)
    print(f"\nSaved clustering results to {results_dir / 'clustering_comparison.csv'}")
    return results_df


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset, split_dataset
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections
    from src.features.feature_extraction import get_feature_matrices

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)
    splits = split_dataset(df)
    features = get_feature_matrices(splits["train"], splits["val"], splits["test"])

    results = run_all_clustering(
        features["flat"]["X_train"],
        features["section"]["X_train"],
        features["labels"]["y_train"],
        features["labels"]["label_encoder"],
    )
    print("\n" + results.to_string(index=False))
