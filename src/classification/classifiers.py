"""
Classification module - train classifiers across multiple feature sets.
"""

import json
import numpy as np
import pandas as pd
from scipy.sparse import issparse, vstack
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:
    SMOTE = None

from src.utils.evaluation import compute_classification_metrics


def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU detected: {torch.cuda.get_device_name(0)} -- using XGBoost GPU training")
            return True
    except ImportError:
        pass
    return False


def _make_xgb_classifier(device="cpu", **kwargs):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        min_child_weight=5, random_state=42,
        device=device, tree_method="hist",
        verbosity=0, **kwargs
    )


def get_classifiers(mode=None):
    use_gpu = _detect_gpu()

    if mode is None:
        if use_gpu:
            print("")
            print("  ============================================")
            print("  Best model: HistGBT on combined features")
            print("  Previous best: 0.810 macro-F1, 83.6% accuracy (sklearn CPU)")
            print("  XGBoost GPU gets ~0.789 macro-F1 but trains 10x faster")
            print("  ============================================")
            print("")
            print("  Training mode options:")
            print("")
            print("    1. GPU only (recommended for quick runs)")
            print("       - Trains XGBoost HistGBT on GPU (~1 min total)")
            print("       - Only 1 classifier but its our best architecture")
            print("       - Expected: ~0.78-0.79 macro-F1 on combined features")
            print("")
            print("    2. CPU only (recommended for final report numbers)")
            print("       - Trains RandomForest + sklearn HistGBT + LinearSVC")
            print("       - sklearn HistGBT historically gives best results (~0.81)")
            print("       - Takes ~10-15 min, all on CPU")
            print("")
            print("    3. Both GPU + CPU (most comprehensive)")
            print("       - Runs all 4 classifiers: XGBoost(GPU) + RF + HistGBT + SVC(CPU)")
            print("       - Best for comparing GPU vs CPU performance in the report")
            print("       - Takes ~10-15 min (CPU classifiers are the bottleneck)")
            print("")
            pick = input("  Enter 1, 2, or 3: ").strip()
            if pick == "1":
                mode = "gpu"
            elif pick == "2":
                mode = "cpu"
            else:
                mode = "both"
        else:
            print("  No GPU detected -- running all classifiers on CPU")
            mode = "cpu"

    clfs = {}

    if mode in ("cpu", "both"):
        clfs["RandomForest"] = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clfs["HistGBT_sklearn"] = HistGradientBoostingClassifier(
            max_iter=500, max_depth=8, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
        clfs["LinearSVC"] = Pipeline([
            ("scaler", MaxAbsScaler()),
            ("svc", LinearSVC(max_iter=50000, class_weight="balanced", random_state=42)),
        ])

    if mode in ("gpu", "both") and use_gpu:
        try:
            clfs["HistGBT"] = _make_xgb_classifier(device="cuda")
        except Exception:
            print("  XGBoost GPU failed, skipping")
            if "HistGBT_sklearn" not in clfs:
                clfs["HistGBT_sklearn"] = HistGradientBoostingClassifier(
                    max_iter=500, max_depth=8, learning_rate=0.1,
                    min_samples_leaf=5, random_state=42
                )

    if mode == "cpu" and not use_gpu:
        # no GPU available, just use sklearn HistGBT
        if "HistGBT_sklearn" not in clfs:
            clfs["HistGBT_sklearn"] = HistGradientBoostingClassifier(
                max_iter=500, max_depth=8, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            )

    print(f"  Running classifiers: {', '.join(clfs.keys())}")
    return clfs


def _apply_smote(X_train, y_train):
    if SMOTE is None:
        return X_train, y_train
    min_count = pd.Series(y_train).value_counts().min()
    k = min(5, min_count - 1)
    if k < 1:
        return X_train, y_train
    sm = SMOTE(random_state=42, k_neighbors=k)
    return sm.fit_resample(X_train, y_train)


_smote_skip = {"HistGBT", "HistGBT_sklearn"}


def train_and_evaluate(X_train, y_train, X_test, y_test, clf, name="", use_smote=True):
    is_ensemble = isinstance(clf, VotingClassifier)
    if use_smote and not is_ensemble and name not in _smote_skip:
        X_train, y_train = _apply_smote(X_train, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["classifier"] = name
    return metrics, y_pred


def get_tuning_grids():
    return {
        "RandomForest": [
            {"n_estimators": 300, "max_depth": None},
            {"n_estimators": 500, "max_depth": None},
        ],
        "HistGBT": [
            {"max_depth": 8, "learning_rate": 0.1},
            {"max_depth": 10, "learning_rate": 0.05},
        ],
        "HistGBT_sklearn": [
            {"max_iter": 500, "max_depth": 8, "learning_rate": 0.1, "min_samples_leaf": 5},
            {"max_iter": 500, "max_depth": 10, "learning_rate": 0.05, "min_samples_leaf": 5},
            {"max_iter": 800, "max_depth": 8, "learning_rate": 0.05, "min_samples_leaf": 3},
            {"max_iter": 800, "max_depth": 12, "learning_rate": 0.03, "min_samples_leaf": 5},
        ],
        "LinearSVC": [
            {"svc__C": 0.5},
            {"svc__C": 1.0},
            {"svc__C": 2.0},
        ],
    }


def _ensure_non_negative(X):
    if hasattr(X, "tocsr"):
        X_nn = X.tocsr(copy=True)
        X_nn.data = np.clip(X_nn.data, 0, None)
        X_nn.eliminate_zeros()
        return X_nn
    return np.clip(X, 0, None)


def _combine_splits(X_train, y_train, X_val=None, y_val=None):
    if X_val is None or y_val is None:
        return X_train, y_train
    if issparse(X_train):
        X_comb = vstack([X_train, X_val], format="csr")
    else:
        X_comb = np.vstack([X_train, X_val])
    y_comb = np.concatenate([y_train, y_val])
    return X_comb, y_comb


def _prepare_inputs_for_model(name, X_train, X_eval, clf=None):
    """Convert to dense for models that need it. XGBoost handles sparse fine."""
    if name == "HistGBT_sklearn":
        X_tr = X_train.toarray() if issparse(X_train) else np.asarray(X_train)
        X_ev = X_eval.toarray() if issparse(X_eval) else np.asarray(X_eval)
        return X_tr, X_ev
    return X_train, X_eval


def _score_tuple(m):
    return (m["macro_f1"], m["balanced_accuracy"], m["accuracy"])


def _select_params_on_validation(name, clf, X_train, y_train, X_val, y_val):
    grids = get_tuning_grids().get(name, [{}])
    best_params = {}; best_metrics = None

    for params in grids:
        candidate = clone(clf)
        if params: candidate.set_params(**params)
        X_fit, X_eval = _prepare_inputs_for_model(name, X_train, X_val, clf=candidate)
        metrics, _ = train_and_evaluate(X_fit, y_train, X_eval, y_val, candidate, name, use_smote=False)
        if best_metrics is None or _score_tuple(metrics) > _score_tuple(best_metrics):
            best_params = params
            best_metrics = metrics

    return best_params, best_metrics


def run_all_classifiers(X_train, y_train, X_test, y_test, mode="flat", X_val=None, y_val=None, trained_models=None, classifiers=None):
    if classifiers is None:
        classifiers = get_classifiers()
    results = []
    predictions = {}
    fitted_clfs = {}
    has_val = X_val is not None and y_val is not None

    for name, clf in classifiers.items():
        print(f"  Training {name} ({mode})...", end=" ")

        if has_val:
            sel_params, val_metrics = _select_params_on_validation(
                name, clf, X_train, y_train, X_val, y_val)
            final_clf = clone(clf)
            if sel_params: final_clf.set_params(**sel_params)
            X_final, y_final = _combine_splits(X_train, y_train, X_val, y_val)
            X_fit, X_te = _prepare_inputs_for_model(name, X_final, X_test, clf=final_clf)
            metrics, y_pred = train_and_evaluate(X_fit, y_final, X_te, y_test, final_clf, name)
            metrics["selection_macro_f1"] = val_metrics["macro_f1"]
            metrics["selection_balanced_accuracy"] = val_metrics["balanced_accuracy"]
            metrics["selected_params"] = json.dumps(sel_params, sort_keys=True)
            metrics["used_validation"] = True
            fitted_clfs[name] = final_clf
        else:
            X_fit, X_te = _prepare_inputs_for_model(name, X_train, X_test, clf=clf)
            metrics, y_pred = train_and_evaluate(X_fit, y_train, X_te, y_test, clf, name)
            metrics["selection_macro_f1"] = np.nan
            metrics["selection_balanced_accuracy"] = np.nan
            metrics["selected_params"] = json.dumps({}, sort_keys=True)
            metrics["used_validation"] = False
            fitted_clfs[name] = clf

        metrics["mode"] = mode
        results.append(metrics)
        predictions[name] = y_pred

        if has_val:
            print(f"val_macro-F1={metrics['selection_macro_f1']:.4f}, test_macro-F1={metrics['macro_f1']:.4f}")
        else:
            print(f"macro-F1={metrics['macro_f1']:.4f}")

    # stash fitted models so we can save without retraining
    if trained_models is not None:
        for name, clf in fitted_clfs.items():
            trained_models[f"{mode}_{name}"] = clf

    return pd.DataFrame(results), predictions


def compare_feature_sets(feature_sets, labels):
    combined_results = []
    all_predictions = {}
    trained_models = {}

    # ask user once, reuse for all feature sets
    classifiers = get_classifiers()

    for mode, features in feature_sets.items():
        print(f"\n=== {mode.replace('_', ' ').title()} Classification ===")
        res, preds = run_all_classifiers(
            features["X_train"], labels["y_train"],
            features["X_test"], labels["y_test"],
            mode=mode,
            X_val=features.get("X_val"), y_val=labels.get("y_val"),
            trained_models=trained_models,
            classifiers=classifiers,
        )
        combined_results.append(res)
        all_predictions[mode] = preds

    combined = pd.concat(combined_results, ignore_index=True)
    return combined, all_predictions, trained_models


def save_trained_models(feature_sets, labels):
    """Train all classifiers and save them to disk for quick re-use."""
    import pickle
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for mode, features in feature_sets.items():
        has_val = features.get("X_val") is not None and labels.get("y_val") is not None
        for name, clf in get_classifiers().items():
            if has_val:
                sel_params, _ = _select_params_on_validation(
                    name, clf, features["X_train"], labels["y_train"],
                    features["X_val"], labels["y_val"])
                final_clf = clone(clf)
                if sel_params: final_clf.set_params(**sel_params)
                X_final, y_final = _combine_splits(
                    features["X_train"], labels["y_train"],
                    features["X_val"], labels["y_val"])
            else:
                final_clf = clone(clf)
                X_final, y_final = features["X_train"], labels["y_train"]

            if name not in _smote_skip:
                X_final, y_final = _apply_smote(X_final, y_final)
            X_fit = X_final.toarray() if issparse(X_final) and name in ("HistGBT",) else X_final
            final_clf.fit(X_fit, y_final)

            key = f"{mode}_{name}"
            saved[key] = final_clf

    out_path = models_dir / "trained_classifiers.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(saved, f)
    print(f"Saved {len(saved)} trained models to {out_path}")
    return saved


def compare_flat_vs_section(flat_features, section_features, labels):
    combined, predictions, _ = compare_feature_sets(
        {"flat": flat_features, "section": section_features}, labels)
    return combined, predictions["flat"], predictions["section"]


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset, split_dataset
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections
    from src.features.feature_extraction import get_feature_matrices
    from src.utils.evaluation import save_results_table, plot_comparison_bar, plot_confusion_matrix

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)
    splits = split_dataset(df)
    features = get_feature_matrices(splits["train"], splits["val"], splits["test"])

    feature_sets = {
        name: features[name]
        for name in ("flat", "section", "hybrid")
        if name in features
    }
    combined, predictions = compare_feature_sets(feature_sets, features["labels"])

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
        output_path=None,
    )
