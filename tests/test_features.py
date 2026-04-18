"""Tests for feature extraction."""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from src.features.feature_extraction import (
    build_flat_tfidf, build_section_tfidf, build_hybrid_features,
    encode_labels, _get_handcrafted
)


def _make_sample_dfs():
    """Create small sample DataFrames for testing."""
    data = {
        "Resume_str": [
            "Python developer with 5 years experience in machine learning and data science",
            "Java engineer experienced in cloud computing and AWS services",
            "Data scientist skilled in statistics python and deep learning models",
            "HR manager with recruitment and employee relations experience",
            "Python data analyst with machine learning and statistics background",
            "Senior HR manager with recruitment and employee engagement skills",
            "Python developer with cloud computing and machine learning experience",
            "HR specialist with employee relations and recruitment experience",
            "Python engineer with data science and machine learning skills",
            "HR coordinator with recruitment and employee management focus",
        ],
        "skills_tokens": [
            ["python", "machine", "learning", "data", "science"],
            ["java", "cloud", "computing", "aws"],
            ["statistics", "python", "deep", "learning"],
            ["recruitment", "employee", "relation", "hr"],
            ["python", "machine", "learning", "statistics"],
            ["recruitment", "employee", "engagement", "hr"],
            ["python", "cloud", "computing", "machine"],
            ["employee", "relation", "recruitment", "hr"],
            ["python", "data", "science", "machine"],
            ["recruitment", "employee", "management", "hr"],
        ],
        "experience_tokens": [
            ["developer", "year", "experience", "data"],
            ["engineer", "cloud", "computing", "year"],
            ["scientist", "data", "analysis", "year"],
            ["manager", "hr", "recruitment", "year"],
            ["analyst", "data", "machine", "year"],
            ["manager", "hr", "employee", "year"],
            ["developer", "cloud", "machine", "year"],
            ["specialist", "hr", "employee", "year"],
            ["engineer", "data", "science", "year"],
            ["coordinator", "hr", "recruitment", "year"],
        ],
        "education_tokens": [
            ["bachelor", "computer", "science"],
            ["master", "engineering", "computer"],
            ["phd", "statistics", "science"],
            ["bachelor", "business", "administration"],
            ["bachelor", "computer", "science"],
            ["master", "business", "administration"],
            ["bachelor", "computer", "engineering"],
            ["bachelor", "business", "management"],
            ["master", "computer", "science"],
            ["bachelor", "business", "administration"],
        ],
        "years_experience": [5, 3, 7, 10, 4, 6, 8, 5, 3, 7],
        "degree_level": ["bachelors", "masters", "phd", "bachelors", "bachelors",
                         "masters", "bachelors", "bachelors", "masters", "bachelors"],
        "num_skills": [5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        "Category": ["IT", "IT", "IT", "HR", "IT", "HR", "IT", "HR", "IT", "HR"],
    }
    df = pd.DataFrame(data)
    train = df.iloc[:8]
    val = df.iloc[8:9]
    test = df.iloc[9:10]
    return train, val, test


class TestBuildFlatTfidf:
    def test_returns_sparse_matrices(self):
        train, val, test = _make_sample_dfs()
        result = build_flat_tfidf(train, val, test, max_features=100)
        assert issparse(result["X_train"])
        assert issparse(result["X_val"])
        assert issparse(result["X_test"])

    def test_shapes_match(self):
        train, val, test = _make_sample_dfs()
        result = build_flat_tfidf(train, val, test, max_features=100)
        assert result["X_train"].shape[0] == len(train)
        assert result["X_val"].shape[0] == len(val)
        assert result["X_test"].shape[0] == len(test)
        # all should have same number of features
        assert result["X_train"].shape[1] == result["X_val"].shape[1]
        assert result["X_train"].shape[1] == result["X_test"].shape[1]

    def test_vectorizer_saved(self):
        train, val, test = _make_sample_dfs()
        result = build_flat_tfidf(train, val, test, max_features=100)
        assert "vectorizer" in result


class TestBuildSectionTfidf:
    def test_returns_sparse_matrices(self):
        train, val, test = _make_sample_dfs()
        result = build_section_tfidf(train, val, test, max_features_per_section=50)
        assert issparse(result["X_train"])

    def test_includes_handcrafted(self):
        train, val, test = _make_sample_dfs()
        result = build_section_tfidf(train, val, test, max_features_per_section=50)
        # should have more features than just 3 sections of TF-IDF
        # (3 handcrafted features added)
        n_features = result["X_train"].shape[1]
        assert n_features >= 3  # at least the handcrafted features

    def test_shapes_consistent(self):
        train, val, test = _make_sample_dfs()
        result = build_section_tfidf(train, val, test, max_features_per_section=50)
        assert result["X_train"].shape[0] == len(train)
        assert result["X_val"].shape[0] == len(val)
        assert result["X_test"].shape[0] == len(test)
        assert result["X_train"].shape[1] == result["X_test"].shape[1]


class TestBuildHybridFeatures:
    def test_returns_sparse_matrices(self):
        train, val, test = _make_sample_dfs()
        flat = build_flat_tfidf(train, val, test, max_features=100)
        section = build_section_tfidf(train, val, test, max_features_per_section=50)
        hybrid = build_hybrid_features(flat, section, latent_components=4)
        assert issparse(hybrid["X_train"])
        assert issparse(hybrid["X_val"])
        assert issparse(hybrid["X_test"])

    def test_adds_latent_dimensions(self):
        train, val, test = _make_sample_dfs()
        flat = build_flat_tfidf(train, val, test, max_features=100)
        section = build_section_tfidf(train, val, test, max_features_per_section=50)
        hybrid = build_hybrid_features(flat, section, latent_components=4)
        assert hybrid["X_train"].shape[1] > flat["X_train"].shape[1]
        assert hybrid["X_train"].shape[1] > section["X_train"].shape[1]

    def test_branch_metadata_present(self):
        train, val, test = _make_sample_dfs()
        flat = build_flat_tfidf(train, val, test, max_features=100)
        section = build_section_tfidf(train, val, test, max_features_per_section=50)
        hybrid = build_hybrid_features(flat, section, latent_components=4)
        assert "branches" in hybrid
        assert {"flat_lsa", "section_lsa"} == set(hybrid["branches"].keys())


class TestEncodeLabels:
    def test_encodes_correctly(self):
        train, val, test = _make_sample_dfs()
        result = encode_labels(train, val, test)
        assert len(result["y_train"]) == len(train)
        assert len(result["y_val"]) == len(val)
        assert len(result["y_test"]) == len(test)

    def test_label_encoder_inverse(self):
        train, val, test = _make_sample_dfs()
        result = encode_labels(train, val, test)
        le = result["label_encoder"]
        decoded = le.inverse_transform(result["y_train"])
        assert list(decoded) == list(train["Category"])


class TestHandcrafted:
    def test_shape(self):
        train, _, _ = _make_sample_dfs()
        hc = _get_handcrafted(train)
        # 7 base + seniority + optionally 2 NER = 8 or 10 features
        assert hc.shape[0] == len(train)
        assert hc.shape[1] in (8, 10)

    def test_degree_encoding(self):
        train, _, _ = _make_sample_dfs()
        hc = _get_handcrafted(train)
        # first row is bachelors = 4
        assert hc[0, 1] == 4.0
        # second row is masters = 5 (IT engineer)
        assert hc[1, 1] == 5.0
        # third row is phd = 6
        assert hc[2, 1] == 6.0
