# metrics + plots

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

results_dir = Path(__file__).resolve().parents[2] / "results"


def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    mp = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": mf1, "balanced_accuracy": bacc,
            "macro_precision": mp, "macro_recall": mr}


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", output_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14,12))

    if sns is not None:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
    else:
        img = ax.imshow(cm, cmap="Blues")
        fig.colorbar(img, ax=ax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    if output_path is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def plot_comparison_bar(results_df, metric="macro_f1", output_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    classifiers = list(results_df["classifier"].unique())
    modes = list(results_df["mode"].unique())
    pivot = results_df.pivot(index="classifier", columns="mode", values=metric)
    pivot = pivot.reindex(classifiers)

    x = np.arange(len(classifiers))
    w = 0.8 / max(len(modes), 1)
    colors = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B2912F"]

    for idx, mode in enumerate(modes):
        offset = (idx - (len(modes)-1)/2) * w
        vals = pivot[mode].values
        label = mode.replace("_", " ").title()
        ax.bar(x + offset, vals, w, label=label, color=colors[idx % len(colors)])

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Feature Set Comparison: {metric.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if output_path is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"comparison_{metric}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def save_results_table(results_df, filename="classification_comparison.csv"):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    results_df.to_csv(path, index=False)
    print(f"\nSaved results to {path}")
    print(results_df.to_string(index=False, float_format="%.4f"))
