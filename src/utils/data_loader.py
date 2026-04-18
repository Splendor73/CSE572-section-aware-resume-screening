"""
Dataset loader for Resume Screening project.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"


def load_primary_dataset(path=None):
    """Load the resume dataset (2484 resumes, 24 categories)."""
    if path is None:
        path = raw_dir / "UpdatedResumeDataSet.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset\n"
            f"Place CSV in: {raw_dir}"
        )

    df = pd.read_csv(path)

    # fix column names if needed
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("resume_str", "resume_string", "resume_text"):
            col_map[col] = "Resume_str"
        elif lower in ("resume_html",):
            col_map[col] = "Resume_html"
        elif lower in ("category",):
            col_map[col] = "Category"
        elif lower in ("id",):
            col_map[col] = "ID"
    df = df.rename(columns=col_map)

    if "ID" not in df.columns:
        df.insert(0, "ID", range(len(df)))

    print(f"Loaded primary dataset: {len(df)} resumes, {df['Category'].nunique()} categories")
    return df


def load_secondary_dataset(path=None):
    if path is None:
        path = raw_dir / "Resume.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Secondary dataset not found at {path}.\n"
            f"Download: https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured\n"
            f"Place CSV in: {raw_dir}"
        )
    df = pd.read_csv(path)
    print(f"Loaded secondary dataset: {len(df)} records, columns: {list(df.columns)}")
    return df


def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """70/15/15 stratified split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["Category"], random_state=random_state,
    )

    val_frac = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_frac, stratify=train_df["Category"], random_state=random_state,
    )

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}


def get_cv_folds(df, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Category"])):
        yield fold, df.iloc[train_idx], df.iloc[val_idx]


def print_dataset_summary(df):
    print(f"\nDataset: {len(df)} resumes, {df['Category'].nunique()} categories\n")
    dist = df["Category"].value_counts()
    print("Class distribution:")
    for cat, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {cat:<30s} {count:>4d}  ({pct:5.1f}%)")
    print(f"\n  Min class size: {dist.min()}, Max class size: {dist.max()}")


def save_splits(splits, output_dir=None):
    if output_dir is None:
        output_dir = processed_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        out = output_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"Saved {name} split ({len(df)} rows) to {out}")


def load_splits(input_dir=None):
    if input_dir is None:
        input_dir = processed_dir
    input_dir = Path(input_dir)
    splits = {}
    for name in ("train", "val", "test"):
        p = input_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}. Run split_dataset() first.")
        splits[name] = pd.read_csv(p)
    print(f"Loaded splits — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    return splits


if __name__ == "__main__":
    df = load_primary_dataset()
    print_dataset_summary(df)
    splits = split_dataset(df)
    save_splits(splits)
