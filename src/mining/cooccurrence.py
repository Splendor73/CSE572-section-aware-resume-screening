# networkx skill graph

import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(__file__).resolve().parents[2] / "results"


def build_cooccurrence_graph(df, min_cooccurrence=5, top_skills=200):
    # count skills
    skill_counts = Counter()
    for toks in df["skills_tokens"]:
        if isinstance(toks, list):
            skill_counts.update(set(toks))
    top_set = set(s for s, _ in skill_counts.most_common(top_skills))

    # count pairs
    cooccur = Counter()
    for toks in df["skills_tokens"]:
        if isinstance(toks, list):
            skills = sorted(set(toks) & top_set)
            for a, b in combinations(skills, 2):
                cooccur[(a,b)] += 1

    G = nx.Graph()
    for (a, b), cnt in cooccur.items():
        if cnt >= min_cooccurrence:
            G.add_edge(a, b, weight=cnt)

    for node in G.nodes():
        G.nodes[node]["frequency"] = skill_counts.get(node, 0)

    print(f"Co-occurrence graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_top_skills(G, top_n=30):
    centrality = nx.degree_centrality(G)
    top = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top


def save_graph_viz(G, output_path=None, top_n=50):
    if G.number_of_nodes() == 0:
        print("Empty graph, skipping visualization.")
        return

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
    sub = G.subgraph(top_nodes)

    fig, ax = plt.subplots(figsize=(14,12))
    pos = nx.spring_layout(sub, k=2, seed=42, iterations=50)  # layout

    node_sizes = [degrees.get(n, 1) * 30 for n in sub.nodes()]
    edge_weights = [sub[u][v].get("weight", 1) for u, v in sub.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [w/max_w * 3 for w in edge_weights]

    nx.draw_networkx_edges(sub, pos, alpha=0.3, width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_size=node_sizes,
                           node_color="#5DA5DA", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)

    ax.set_title(f"Skill Co-occurrence Network (top {len(sub.nodes())} skills)")
    ax.axis("off")
    plt.tight_layout()

    if output_path is None:
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / "cooccurrence_graph.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved graph visualization to {output_path}")


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)

    G = build_cooccurrence_graph(df, min_cooccurrence=5, top_skills=200)

    print("\nTop 30 skills by degree centrality:")
    for skill, cent in get_top_skills(G, top_n=30):
        print(f"  {skill:<30s} centrality={cent:.4f}")

    save_graph_viz(G, top_n=50)
