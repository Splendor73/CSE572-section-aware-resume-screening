import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.data_loader import (
    load_primary_dataset,
    split_dataset,
    get_cv_folds
)

def test_split_dataset():
    # Create a dummy dataframe with enough categories for stratify
    data = {
        "ID": range(20),
        "Category": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
                     "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
    }
    df = pd.DataFrame(data)
    
    splits = split_dataset(df, test_size=0.2, val_size=0.2, random_state=42)
    
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    
    # Check sizes (20 total, 0.2 test = 4, 0.2 val = 4, 0.6 train = 12)
    assert len(splits["test"]) == 4
    assert len(splits["val"]) == 4
    assert len(splits["train"]) == 12
    
    # Check stratification: test should have 2 A's and 2 B's
    assert splits["test"]["Category"].value_counts()["A"] == 2
    assert splits["test"]["Category"].value_counts()["B"] == 2

def test_get_cv_folds():
    data = {
        "ID": range(10),
        "Category": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    }
    df = pd.DataFrame(data)
    
    folds = list(get_cv_folds(df, n_splits=5, random_state=42))
    
    assert len(folds) == 5
    for fold_num, train_df, val_df in folds:
        assert len(val_df) == 2
        assert len(train_df) == 8
