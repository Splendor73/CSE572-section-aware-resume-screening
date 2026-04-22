# tfidf + sbert + hybrid stuff

import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
except (ModuleNotFoundError, OSError):
    _nlp = None


def _tokens_to_text(tokens_series):
    return tokens_series.apply(lambda toks: " ".join(toks) if isinstance(toks, list) else "")


def build_flat_tfidf(train_df, val_df, test_df, max_features=8000):
    vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True,
                          min_df=2, ngram_range=(1,2), stop_words="english")
    X_train = vec.fit_transform(train_df["Resume_str"].fillna(""))
    X_val   = vec.transform(val_df["Resume_str"].fillna(""))
    X_test  = vec.transform(test_df["Resume_str"].fillna(""))
    print(f"Flat TF-IDF: {X_train.shape[1]} features")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "vectorizer": vec}


def build_section_tfidf(train_df, val_df, test_df, max_features_per_section=3000):
    sections = ["skills_tokens", "experience_tokens", "education_tokens"]
    vectorizers = {}
    train_parts = []; val_parts = []; test_parts = []

    for sec in sections:
        v = TfidfVectorizer(
            max_features=max_features_per_section, sublinear_tf=True, min_df=2, ngram_range=(1, 2),
        )

        tr_text = _tokens_to_text(train_df[sec])
        va_text = _tokens_to_text(val_df[sec])
        te_text = _tokens_to_text(test_df[sec])

        X_tr = v.fit_transform(tr_text)
        X_va = v.transform(va_text)
        X_te = v.transform(te_text)

        train_parts.append(X_tr)
        val_parts.append(X_va)
        test_parts.append(X_te)
        vectorizers[sec] = v
        print(f"  {sec}: {X_tr.shape[1]} features")

    # add handcrafted
    for split_df, parts in [(train_df, train_parts), (val_df, val_parts), (test_df, test_parts)]:
        hc = _get_handcrafted(split_df)
        parts.append(csr_matrix(hc))

    X_train = hstack(train_parts, format="csr")
    X_val = hstack(val_parts, format="csr")
    X_test = hstack(test_parts, format="csr")

    print(f"Section-aware TF-IDF: {X_train.shape[1]} total features (incl. handcrafted)")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "vectorizers": vectorizers}


def _fit_lsa_branch(X_train, X_val, X_test, n_components=128, branch_name="feature"):
    n_components = max(1, min(n_components, min(X_train.shape[0]-1, X_train.shape[1]-1)))
    svd  = TruncatedSVD(n_components=n_components, random_state=42)
    norm = Normalizer(copy=False)
    Xtr  = norm.fit_transform(svd.fit_transform(X_train))
    Xva  = norm.transform(svd.transform(X_val))
    Xte  = norm.transform(svd.transform(X_test))
    ev = float(svd.explained_variance_ratio_.sum())
    print(f"  {branch_name} latent branch: {n_components} dims, explained variance={ev:.3f}")
    return {"X_train": Xtr, "X_val": Xva, "X_test": Xte,
            "svd": svd, "normalizer": norm, "explained_variance": ev}


def build_hybrid_features(flat_features, section_features, latent_components=128):
    flat_lsa = _fit_lsa_branch(
        flat_features["X_train"], flat_features["X_val"], flat_features["X_test"],
        n_components=latent_components, branch_name="flat",
    )
    section_lsa = _fit_lsa_branch(
        section_features["X_train"], section_features["X_val"], section_features["X_test"],
        n_components=latent_components, branch_name="section",
    )

    X_train = hstack([flat_features["X_train"], section_features["X_train"],
                       csr_matrix(flat_lsa["X_train"]), csr_matrix(section_lsa["X_train"])], format="csr")
    X_val = hstack([flat_features["X_val"], section_features["X_val"],
                     csr_matrix(flat_lsa["X_val"]), csr_matrix(section_lsa["X_val"])], format="csr")
    X_test = hstack([flat_features["X_test"], section_features["X_test"],
                      csr_matrix(flat_lsa["X_test"]), csr_matrix(section_lsa["X_test"])], format="csr")

    print(f"Hybrid features: {X_train.shape[1]} total features (flat + section + latent branches)")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "branches": {"flat_lsa": flat_lsa, "section_lsa": section_lsa},
    }


def _detect_seniority(text):
    if not isinstance(text, str):
        return 0
    t = text.lower()
    if re.search(r"\b(vp|vice\s*president|c[eo]o|cto|cfo)\b", t): return 7
    if re.search(r"\b(director)\b", t):                              return 6
    if re.search(r"\b(manager|management)\b", t):                   return 5
    if re.search(r"\b(lead|principal|staff)\b", t):                 return 4
    if re.search(r"\b(senior|sr\.?)\b", t):                         return 3
    if re.search(r"\b(associate|mid[- ]level)\b", t):               return 2
    if re.search(r"\b(junior|jr\.?|entry[- ]level)\b", t):         return 1
    if re.search(r"\b(intern|trainee|apprentice)\b", t):            return 0
    return 0


def _count_ner_entities(text):
    if _nlp is None or not isinstance(text, str) or len(text)==0:
        return 0, 0
    doc = _nlp(text[:5000])
    orgs = len(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    gpes = len(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))
    return orgs, gpes


def _get_handcrafted(df):
    deg = {"unknown": 0, "high_school": 1, "diploma": 2,
           "associates": 3, "bachelors": 4, "masters": 5, "phd": 6}

    yrs   = df["years_experience"].fillna(0).values.astype(float)
    deglv = df["degree_level"].map(deg).fillna(0).values.astype(float)
    nsk   = df["num_skills"].fillna(0).values.astype(float)
    sk_len  = df["skills_tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).values.astype(float)
    ex_len  = df["experience_tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).values.astype(float)
    edu_len = df["education_tokens"].apply(lambda x: len(x) if isinstance(x, list) else 0).values.astype(float)
    res_len = df["Resume_str"].fillna("").apply(len).values.astype(float)

    base = np.column_stack([yrs, deglv, nsk, sk_len, ex_len, edu_len, res_len])

    seniority = df["Resume_str"].fillna("").apply(_detect_seniority).values.astype(float).reshape(-1,1)

    if _nlp is not None:
        texts = df["Resume_str"].fillna("").apply(lambda x: x[:5000]).tolist()
        org_counts = []
        gpe_counts = []
        for doc in _nlp.pipe(texts, batch_size=64):
            org_counts.append(len(set(e.text for e in doc.ents if e.label_ == "ORG")))
            gpe_counts.append(len(set(e.text for e in doc.ents if e.label_ == "GPE")))
        org_counts = np.array(org_counts, dtype=float).reshape(-1,1)
        gpe_counts = np.array(gpe_counts, dtype=float).reshape(-1,1)
        return np.hstack([base, seniority, org_counts, gpe_counts])

    return np.hstack([base, seniority])


def _get_device():
    import torch
    if torch.cuda.is_available():
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Using GPU: Apple MPS (Metal)")
        return "mps"
    else:
        print("  Using CPU (no GPU detected)")
        return "cpu"


def build_semantic_features(train_df, val_df, test_df, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        print("sentence-transformers not available, skipping semantic features")
        return None

    import warnings, logging
    logging.getLogger("safetensors").setLevel(logging.ERROR)
    device = _get_device()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer(model_name, device=device)
    sections = ["skills_tokens", "experience_tokens", "education_tokens"]

    parts = {"train": [], "val": [], "test": []}

    for sec in sections:
        for sname, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            texts = _tokens_to_text(sdf[sec]).tolist()
            texts = [t if len(t) > 0 else "empty" for t in texts]
            emb = model.encode(texts, show_progress_bar=False, batch_size=128)
            parts[sname].append(emb)
        print(f"  {sec}: {emb.shape[1]} dims")

    # also encode full resume text
    for sname, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        texts = sdf["Resume_str"].fillna("").tolist()
        texts = [t[:512] if len(t)>0 else "empty" for t in texts]
        emb = model.encode(texts, show_progress_bar=False, batch_size=128)
        parts[sname].append(emb)
    print(f"  full_resume: {emb.shape[1]} dims")

    for sname, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        hc = _get_handcrafted(sdf)
        parts[sname].append(hc)

    X_train = np.hstack(parts["train"])
    X_val = np.hstack(parts["val"])
    X_test = np.hstack(parts["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Semantic features: {X_train.shape[1]} total dims (4 sections + handcrafted, scaled)")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "scaler": scaler}


def encode_labels(train_df, val_df, test_df):
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["Category"])
    y_val = le.transform(val_df["Category"])
    y_test = le.transform(test_df["Category"])
    print(f"Labels: {len(le.classes_)} classes")
    return {"y_train": y_train, "y_val": y_val, "y_test": y_test, "label_encoder": le}


def _apply_feature_selection(feat_dict, y_train, k=3000, name="features"):
    k2  = min(k, feat_dict["X_train"].shape[1])
    sel = SelectKBest(chi2, k=k2)

    Xtr = sel.fit_transform(feat_dict["X_train"], y_train)
    Xva = sel.transform(feat_dict["X_val"])
    Xte = sel.transform(feat_dict["X_test"])

    print(f"  {name}: {feat_dict['X_train'].shape[1]} -> {Xtr.shape[1]} features (chi2)")
    d = dict(feat_dict)
    d["X_train"] = Xtr
    d["X_val"]   = Xva
    d["X_test"]  = Xte
    d["selector"] = sel
    return d


def _build_combined(flat_features, semantic_features):
    from scipy.sparse import issparse as _issparse
    def _to_dense(X):
        return X.toarray() if _issparse(X) else np.asarray(X)

    X_train = np.hstack([_to_dense(flat_features["X_train"]), semantic_features["X_train"]])
    X_val = np.hstack([_to_dense(flat_features["X_val"]), semantic_features["X_val"]])
    X_test = np.hstack([_to_dense(flat_features["X_test"]), semantic_features["X_test"]])
    print(f"Combined features: {X_train.shape[1]} dims (flat TF-IDF + semantic)")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test}


def get_feature_matrices(train_df, val_df, test_df):
    print("\n--- Building flat TF-IDF ---")
    flat = build_flat_tfidf(train_df, val_df, test_df)

    print("\n--- Building section-aware TF-IDF ---")
    section = build_section_tfidf(train_df, val_df, test_df)

    labels = encode_labels(train_df, val_df, test_df)

    print("\n--- Feature selection (chi-squared) ---")
    flat = _apply_feature_selection(flat, labels["y_train"], k=4000, name="flat")
    section = _apply_feature_selection(section, labels["y_train"], k=4000, name="section")

    print("\n--- Building hybrid latent-semantic features ---")
    hybrid = build_hybrid_features(flat, section)

    print("\n--- Building semantic (Sentence Transformer) features ---")
    semantic = build_semantic_features(train_df, val_df, test_df)

    result = {"flat": flat, "section": section, "hybrid": hybrid, "labels": labels}
    if semantic is not None:
        result["semantic"] = semantic
        print("\n--- Building combined (TF-IDF + Semantic) features ---")
        result["combined"] = _build_combined(flat, semantic)
    return result


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset, split_dataset
    from src.parsing.html_parser import parse_all_resumes
    from src.agents.section_agents import process_all_sections

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)

    splits = split_dataset(df)
    features = get_feature_matrices(splits["train"], splits["val"], splits["test"])

    print(f"\nFlat:    train={features['flat']['X_train'].shape}, val={features['flat']['X_val'].shape}, test={features['flat']['X_test'].shape}")
    print(f"Section: train={features['section']['X_train'].shape}, val={features['section']['X_val'].shape}, test={features['section']['X_test'].shape}")
    print(f"Labels:  train={features['labels']['y_train'].shape}, val={features['labels']['y_val'].shape}, test={features['labels']['y_test'].shape}")

    flat_zero = (features["flat"]["X_train"].sum(axis=1) == 0).sum()
    sec_zero = (features["section"]["X_train"].sum(axis=1) == 0).sum()
    print(f"\nZero-rows: flat={flat_zero}, section={sec_zero}")
