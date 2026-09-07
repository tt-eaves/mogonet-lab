import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


DATA_DIR = "ROSMAP"
OUT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(omic_id: str):
    tr = np.loadtxt(os.path.join(DATA_DIR, f"{omic_id}_tr.csv"), delimiter=",")
    te = np.loadtxt(os.path.join(DATA_DIR, f"{omic_id}_te.csv"), delimiter=",")
    X = np.vstack([tr, te])

    y_tr = np.loadtxt(os.path.join(DATA_DIR, "labels_tr.csv"), delimiter=",").astype(int)
    y_te = np.loadtxt(os.path.join(DATA_DIR, "labels_te.csv"), delimiter=",").astype(int)
    y = np.concatenate([y_tr, y_te])
    return X, y


def plot_pca(X, y, omic_name: str):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "class": y})
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="class",
        palette="Set2",
        s=50,
        alpha=0.9,
    )
    plt.title(f"{omic_name} PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{omic_name.lower()}_pca.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_feature_heatmap(X, y, omic_name: str, top_n: int = 40):
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[::-1][:top_n]
    X_top = X[:, top_idx]

    X_scaled = StandardScaler().fit_transform(X_top)
    order = np.argsort(y)
    X_ordered = X_scaled[order]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        X_ordered,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar=True,
    )
    plt.title(f"{omic_name} top {top_n} variable features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{omic_name.lower()}_top_features.png"), dpi=220, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(X, omic_name: str):
    sim = cosine_similarity(X)
    plt.figure(figsize=(7, 6))
    sns.heatmap(sim, cmap="magma", square=True, cbar_kws={"shrink": 0.9})
    plt.title(f"{omic_name} sample cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{omic_name.lower()}_similarity.png"), dpi=220, bbox_inches="tight")
    plt.close()


def main():
    for omic_id, omic_name in [("1", "mRNA"), ("2", "methDNA"), ("3", "miRNA")]:
        X, y = load_data(omic_id)
        plot_pca(X, y, omic_name)
        plot_top_feature_heatmap(X, y, omic_name)
        plot_similarity_heatmap(X, omic_name)
        print(f"Saved plots for {omic_name} to {OUT_DIR}")


if __name__ == "__main__":
    main()
