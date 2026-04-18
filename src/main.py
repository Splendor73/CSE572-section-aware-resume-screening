"""
Section-Aware Multi-Agent Resume Screening Pipeline
Main entry point.
"""

import argparse
import pickle
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

project_root = Path(__file__).resolve().parents[1]
processed_dir = project_root / "data" / "processed"
results_dir = project_root / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Section-Aware Multi-Agent Resume Screening Pipeline"
    )
    parser.add_argument(
        "--stage", choices=["parsing", "features", "mining", "classification", "predict", "all"],
        default="all", help="Pipeline stage to run (default: all)",
    )
    return parser.parse_args()


def run_parsing():
    print("\n" + "="*60)
    print("Stage 1: Parsing resumes and extracting sections...")
    print("="*60)

    from src.utils.data_loader import load_primary_dataset, split_dataset, save_splits
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)

    splits = split_dataset(df)
    save_splits(splits)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(processed_dir / "parsed_full.pkl")
    for name, split_df in splits.items():
        split_df.to_pickle(processed_dir / f"{name}_parsed.pkl")

    print("Parsing complete.")
    return splits


def run_features(splits=None):
    print("\n" + "="*60)
    print("Stage 2: Extracting features...")
    print("="*60)

    from src.features.feature_extraction import get_feature_matrices
    import pandas as pd

    if splits is None:
        splits = {
            name: pd.read_pickle(processed_dir / f"{name}_parsed.pkl")
            for name in ("train", "val", "test")
        }

    features = get_feature_matrices(splits["train"], splits["val"], splits["test"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("Feature extraction complete.")
    return features


def run_mining(splits=None):
    print("\n" + "="*60)
    print("Stage 3: Mining patterns...")
    print("="*60)

    import pandas as pd
    from src.mining.association_rules import build_skill_transactions, mine_rules, save_rules
    from src.mining.cooccurrence import build_cooccurrence_graph, get_top_skills, save_graph_viz

    if splits is None:
        full_df = pd.read_pickle(processed_dir / "parsed_full.pkl")
    else:
        full_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)

    print("\n--- Association Rules ---")
    transactions = build_skill_transactions(full_df, top_n=200)
    rules = mine_rules(transactions, min_support=0.05)
    if len(rules) > 0:
        save_rules(rules)

    print("\n--- Co-occurrence Network ---")
    G = build_cooccurrence_graph(full_df, min_cooccurrence=5, top_skills=200)
    top = get_top_skills(G, top_n=20)
    print("Top 20 skills by degree centrality:")
    for skill, cent in top:
        print(f"  {skill:<30s} {cent:.4f}")
    save_graph_viz(G, top_n=50)

    print("\n--- Clustering ---")
    try:
        with open(processed_dir / "features.pkl", "rb") as f:
            features = pickle.load(f)
    except FileNotFoundError:
        features = run_features(splits)

    from src.clustering.cluster import run_all_clustering
    run_all_clustering(
        features["flat"]["X_train"], features["section"]["X_train"],
        features["labels"]["y_train"], features["labels"]["label_encoder"],
    )
    print("Mining complete.")


def run_classification(features=None):
    print("\n" + "="*60)
    print("Stage 4: Running classification...")
    print("="*60)

    if features is None:
        with open(processed_dir / "features.pkl", "rb") as f:
            features = pickle.load(f)

    from src.classification.classifiers import compare_feature_sets
    from src.utils.evaluation import save_results_table, plot_comparison_bar, plot_confusion_matrix

    feature_sets = {
        name: features[name]
        for name in ("flat", "section", "hybrid", "semantic", "combined")
        if name in features
    }
    combined, predictions, trained_models = compare_feature_sets(feature_sets, features["labels"])

    # save trained models for quick re-use (already fitted, no retraining)
    import pickle as _pkl
    models_dir = processed_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "trained_classifiers.pkl", "wb") as f:
        _pkl.dump(trained_models, f)
    print(f"Saved {len(trained_models)} trained models to {models_dir / 'trained_classifiers.pkl'}")

    save_results_table(combined)
    plot_comparison_bar(combined, metric="macro_f1")
    plot_comparison_bar(combined, metric="accuracy")

    best_row = combined.sort_values("macro_f1", ascending=False).iloc[0]
    best_name = best_row["classifier"]
    best_mode = best_row["mode"]
    le = features["labels"]["label_encoder"]
    plot_confusion_matrix(
        features["labels"]["y_test"],
        predictions[best_mode][best_name],
        le.classes_,
        title=f"Best Model: {best_mode.title()} {best_name}",
    )
    print("Classification complete.")


def run_predict():
    """Try to load saved weights. If missing, tell user their options."""
    print("\n" + "="*60)
    print("Quick predict: looking for saved model weights...")
    print("="*60)

    models_path = processed_dir / "models" / "trained_classifiers.pkl"
    features_path = processed_dir / "features.pkl"

    # check features first
    if not features_path.exists():
        print(f"\n[ERROR] Feature file not found: {features_path}")
        print("You need to run the full pipeline first:")
        print("  python src/main.py --stage all")
        return

    # check for saved model weights
    if not models_path.exists():
        print(f"\n[WARNING] No saved model weights found at:")
        print(f"  {models_path}\n")
        print("You have two options:")
        print("  1. Download the pre-trained weights and place them at:")
        print(f"     {models_path}")
        print("")
        print("  2. Train from scratch by running:")
        print("     python src/main.py --stage classification")
        print("     (this will train all models and save weights automatically)")
        return

    from scipy.sparse import issparse
    from src.utils.evaluation import compute_classification_metrics

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    with open(models_path, "rb") as f:
        models = pickle.load(f)

    y_test = features["labels"]["y_test"]
    le = features["labels"]["label_encoder"]

    from src.utils.evaluation import save_results_table, plot_comparison_bar, plot_confusion_matrix
    import pandas as pd

    clf_names = {"RandomForest", "HistGBT", "HistGBT_sklearn", "LinearSVC"}
    print(f"Found {len(models)} saved models, running predictions...\n")
    all_results = []
    all_predictions = {}
    for key, clf in models.items():
        # find which classifier name is in the key
        name = next((c for c in clf_names if key.endswith("_" + c)), None)
        if name is None:
            continue
        mode = key[: -(len(name) + 1)]
        if mode not in features:
            continue
        X_test = features[mode]["X_test"]
        if name in ("HistGBT", "HistGBT_sklearn") and issparse(X_test):
            X_test = X_test.toarray()
        y_pred = clf.predict(X_test)
        m = compute_classification_metrics(y_test, y_pred)
        m["classifier"] = name
        m["mode"] = mode
        all_results.append(m)
        all_predictions[key] = y_pred
        print(f"  {key:<35s} acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")

    # save results, plots, confusion matrix
    combined = pd.DataFrame(all_results)
    save_results_table(combined)
    plot_comparison_bar(combined, metric="macro_f1")
    plot_comparison_bar(combined, metric="accuracy")

    best_row = combined.sort_values("macro_f1", ascending=False).iloc[0]
    best_key = f"{best_row['mode']}_{best_row['classifier']}"
    plot_confusion_matrix(
        y_test, all_predictions[best_key], le.classes_,
        title=f"Best Model: {best_row['mode'].title()} {best_row['classifier']}",
    )

    print("\nDone. Results saved to results/")


def main():
    args = parse_args()
    splits = None; features = None

    if args.stage in ("parsing", "all"):
        splits = run_parsing()
    if args.stage == "features":
        features = run_features(splits)
    elif args.stage == "all":
        features_path = processed_dir / "features.pkl"
        if features_path.exists():
            print("\n" + "="*60)
            print(f"Saved features found at: {features_path}")
            print("")
            print("What would you like to do?")
            print("  1. Use saved features (skip extraction, much faster)")
            print("  2. Re-extract features from scratch")
            print("")
            feat_choice = input("Enter 1 or 2: ").strip()
            if feat_choice == "2":
                features = run_features(splits)
            else:
                print("Loading cached features...")
                with open(features_path, "rb") as f:
                    features = pickle.load(f)
                print("Loaded.")
        else:
            features = run_features(splits)
    if args.stage in ("mining", "all"):
        run_mining(splits)
    if args.stage in ("classification", "all"):
        models_path = processed_dir / "models" / "trained_classifiers.pkl"
        features_path = processed_dir / "features.pkl"

        if models_path.exists() and features_path.exists():
            print("\n" + "="*60)
            print("Saved model weights found at:")
            print(f"  {models_path}")
            print("")
            print("What would you like to do?")
            print("  1. Use saved weights (quick, no retraining)")
            print("  2. Train again from scratch (takes 5-10 min)")
            print("")
            choice = input("Enter 1 or 2: ").strip()

            if choice == "1":
                run_predict()
            else:
                run_classification(features)
        else:
            run_classification(features)

    if args.stage == "predict":
        run_predict()

    print("\n" + "="*60)
    print("Pipeline complete. Results saved to results/")
    if args.stage in ("classification", "all"):
        print(f"Model weights saved to: data/processed/models/trained_classifiers.pkl")
        print(f"Next time, run quick predictions without retraining:")
        print(f"  python src/main.py --stage predict")
    print("="*60)


if __name__ == "__main__":
    main()
