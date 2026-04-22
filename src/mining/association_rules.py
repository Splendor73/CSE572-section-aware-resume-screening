# fp growth on skills

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pathlib import Path

results_dir = Path(__file__).resolve().parents[2] / "results"


def build_skill_transactions(df, top_n=300):
    from collections import Counter

    all_skills = Counter()
    for tokens in df["skills_tokens"]:
        if isinstance(tokens, list):
            all_skills.update(set(tokens))

    top = [s for s, _ in all_skills.most_common(top_n)]

    rows = []
    for tokens in df["skills_tokens"]:
        token_set = set(tokens) if isinstance(tokens, list) else set()
        rows.append({s: (s in token_set) for s in top})

    tx_df = pd.DataFrame(rows)
    print(f"Skill transactions: {tx_df.shape[0]} resumes x {tx_df.shape[1]} skills")
    return tx_df


def mine_rules(transactions_df, min_support=0.05, min_threshold=1.0):
    frequent = fpgrowth(transactions_df, min_support=min_support, use_colnames=True)
    print(f"Frequent itemsets: {len(frequent)}")

    if len(frequent) == 0:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False)
    print(f"Association rules: {len(rules)}")
    return rules


def mine_rules_by_category(df, top_categories=5, min_support=0.1):
    top_cats = df["Category"].value_counts().head(top_categories).index.tolist()
    cat_rules = {}
    for cat in top_cats:
        print(f"\n  Category: {cat}")
        cat_df = df[df["Category"] == cat]
        tx = build_skill_transactions(cat_df, top_n=100)
        r = mine_rules(tx, min_support=min_support)
        if len(r) > 0:
            cat_rules[cat] = r
            print(f"    Top rule: {list(r.iloc[0]['antecedents'])} -> {list(r.iloc[0]['consequents'])} (lift={r.iloc[0]['lift']:.2f})")
    return cat_rules


def save_rules(rules, filename="association_rules_top.csv"):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename

    if len(rules) == 0:
        print("No rules to save.")
        return

    save_df = rules.copy()
    save_df["antecedents"] = save_df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    save_df["consequents"] = save_df["consequents"].apply(lambda x: ", ".join(sorted(x)))
    save_df.head(50).to_csv(path, index=False)
    print(f"Saved top 50 rules to {path}")


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)

    print("\n=== Global Association Rules ===")
    transactions = build_skill_transactions(df, top_n=200)
    rules = mine_rules(transactions, min_support=0.05)
    if len(rules) > 0:
        print("\nTop 20 rules by lift:")
        for _, r in rules.head(20).iterrows():
            ant = ", ".join(sorted(r["antecedents"]))
            con = ", ".join(sorted(r["consequents"]))
            print(f"  {ant} -> {con}  (support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.2f})")
        save_rules(rules)

    print("\n=== Per-Category Rules ===")
    cat_rules = mine_rules_by_category(df)
