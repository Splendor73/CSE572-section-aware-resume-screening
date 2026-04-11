"""
Dataset loader for the Resume Screening project.

Primary dataset: Kaggle Resume Dataset (2,484 resumes, 24 job categories)
    - https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Secondary dataset: Structured Resume Dataset (vocab expansion only)
    - https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Primary dataset
# ---------------------------------------------------------------------------

PRIMARY_FILENAME = "UpdatedResumeDataSet.csv"
SECONDARY_FILENAME = "Resume.csv"


def load_primary_dataset(path=None):
    """Load the primary resume dataset (2,484 resumes with text, HTML, and category).

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. Defaults to data/raw/UpdatedResumeDataSet.csv

    Returns
    -------
    pd.DataFrame
        Columns: ID, Resume_str, Resume_html, Category
    """
    if path is None:
        path = RAW_DIR / PRIMARY_FILENAME

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Primary dataset not found at {path}.\n"
            f"Download it from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset\n"
            f"Place the CSV file in: {RAW_DIR}"
        )

    df = pd.read_csv(path)

    # Standardize column names
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

    # Add ID if missing
    if "ID" not in df.columns:
        df.insert(0, "ID", range(len(df)))

    print(f"Loaded primary dataset: {len(df)} resumes, {df['Category'].nunique()} categories")
    return df


def load_secondary_dataset(path=None):
    """Load the secondary structured resume dataset (vocab expansion only).

    Parameters
    ----------
    path : str or Path, optional
        Path to the CSV file. Defaults to data/raw/Resume.csv

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = RAW_DIR / SECONDARY_FILENAME

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Secondary dataset not found at {path}.\n"
            f"Download it from: https://www.kaggle.com/datasets/suriyaganesh/resume-dataset-structured\n"
            f"Place the CSV file in: {RAW_DIR}"
        )

    df = pd.read_csv(path)
    print(f"Loaded secondary dataset: {len(df)} records, columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """Stratified 70/15/15 train/val/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Category' column.
    test_size : float
    val_size : float
    random_state : int

    Returns
    -------
    dict
        {"train": df, "val": df, "test": df}
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Category"],
        random_state=random_state,
    )

    # val_size is relative to the original dataset, adjust for remaining data
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=train_df["Category"],
        random_state=random_state,
    )

    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}


def get_cv_folds(df, n_splits=5, random_state=42):
    """Generator that yields stratified 5-fold cross-validation splits.

    Yields
    ------
    tuple(pd.DataFrame, pd.DataFrame)
        (train_fold, val_fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["Category"])):
        yield fold, df.iloc[train_idx], df.iloc[val_idx]


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def print_dataset_summary(df):
    """Print class distribution and basic stats."""
    print(f"\nDataset: {len(df)} resumes, {df['Category'].nunique()} categories\n")
    dist = df["Category"].value_counts()
    print("Class distribution:")
    for cat, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {cat:<30s} {count:>4d}  ({pct:5.1f}%)")
    print(f"\n  Min class size: {dist.min()}, Max class size: {dist.max()}")


# ---------------------------------------------------------------------------
# Convenience: save/load processed splits
# ---------------------------------------------------------------------------

def save_splits(splits, output_dir=None):
    """Save train/val/test splits to CSV."""
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {name} split ({len(df)} rows) to {out_path}")


def load_splits(input_dir=None):
    """Load previously saved train/val/test splits."""
    if input_dir is None:
        input_dir = PROCESSED_DIR
    input_dir = Path(input_dir)

    splits = {}
    for name in ("train", "val", "test"):
        path = input_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}. Run split_dataset() first.")
        splits[name] = pd.read_csv(path)
    print(f"Loaded splits — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    return splits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_primary_dataset()
    print_dataset_summary(df)
    splits = split_dataset(df)
    save_splits(splits)
