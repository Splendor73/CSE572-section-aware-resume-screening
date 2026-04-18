"""Tests for mining modules (association rules + co-occurrence)."""

import pytest
import pandas as pd
import networkx as nx
from src.mining.association_rules import build_skill_transactions, mine_rules
from src.mining.cooccurrence import build_cooccurrence_graph, get_top_skills


def _make_sample_df():
    return pd.DataFrame({
        "skills_tokens": [
            ["python", "sql", "machine learning", "tensorflow"],
            ["python", "java", "sql", "docker"],
            ["python", "sql", "pandas", "numpy"],
            ["java", "spring", "sql", "docker"],
            ["python", "machine learning", "tensorflow", "keras"],
            ["sql", "excel", "accounting", "finance"],
            ["python", "sql", "data", "analysis"],
            ["java", "sql", "spring", "kubernetes"],
            ["python", "machine learning", "sql", "numpy"],
            ["excel", "sql", "accounting", "finance"],
        ],
        "Category": ["IT"] * 5 + ["FINANCE"] * 2 + ["IT"] * 2 + ["FINANCE"],
    })


class TestBuildSkillTransactions:
    def test_returns_boolean_dataframe(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=20)
        assert isinstance(transactions, pd.DataFrame)
        assert transactions.dtypes.nunique() == 1  # all same type
        assert len(transactions) == len(df)

    def test_top_n_limits_columns(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=5)
        assert transactions.shape[1] == 5

    def test_correct_values(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=20)
        # first row has python
        assert transactions.iloc[0]["python"] == True
        # first row does not have java
        if "java" in transactions.columns:
            assert transactions.iloc[0]["java"] == False


class TestMineRules:
    def test_returns_dataframe(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=20)
        rules = mine_rules(transactions, min_support=0.2)
        assert isinstance(rules, pd.DataFrame)

    def test_rules_have_expected_columns(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=20)
        rules = mine_rules(transactions, min_support=0.2)
        if len(rules) > 0:
            assert "antecedents" in rules.columns
            assert "consequents" in rules.columns
            assert "lift" in rules.columns
            assert "support" in rules.columns

    def test_high_min_support_gives_fewer_rules(self):
        df = _make_sample_df()
        transactions = build_skill_transactions(df, top_n=20)
        rules_low = mine_rules(transactions, min_support=0.2)
        rules_high = mine_rules(transactions, min_support=0.5)
        assert len(rules_high) <= len(rules_low)


class TestCooccurrenceGraph:
    def test_returns_graph(self):
        df = _make_sample_df()
        G = build_cooccurrence_graph(df, min_cooccurrence=2, top_skills=20)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0

    def test_edges_have_weight(self):
        df = _make_sample_df()
        G = build_cooccurrence_graph(df, min_cooccurrence=2, top_skills=20)
        for u, v, data in G.edges(data=True):
            assert "weight" in data
            assert data["weight"] >= 2

    def test_min_cooccurrence_filters(self):
        df = _make_sample_df()
        G_low = build_cooccurrence_graph(df, min_cooccurrence=1, top_skills=20)
        G_high = build_cooccurrence_graph(df, min_cooccurrence=5, top_skills=20)
        assert G_high.number_of_edges() <= G_low.number_of_edges()

    def test_get_top_skills(self):
        df = _make_sample_df()
        G = build_cooccurrence_graph(df, min_cooccurrence=2, top_skills=20)
        top = get_top_skills(G, top_n=5)
        assert len(top) <= 5
        assert all(isinstance(s, str) for s, _ in top)
        assert all(isinstance(c, float) for _, c in top)
