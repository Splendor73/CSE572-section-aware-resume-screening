# Section-Aware Multi-Agent Resume Screening and Skill Mining

CSE 572 (Data Mining) — Arizona State University, Spring 2026 — Group project.

## Summary

We parse résumé HTML into sections (Skills, Experience, Education), run section-specific “agents” (TF‑IDF, normalization, and light rule-based features), and optionally add SentenceTransformer embeddings (all-MiniLM-L6-v2) concatenated with flat TF‑IDF for classification. The pipeline also runs skill co‑occurrence analysis, FP‑Growth association rules, and clustering (K‑Means, hierarchical, DBSCAN) for exploratory mining. Best reported result on a held-out test set: **HistGradientBoosting on combined (flat TF‑IDF + semantic) features** — about **83.6% accuracy** and **0.81 macro‑F1** over 24 imbalanced job categories (stratified train/validation/test split).

## Team

| Name | ASU ID |
|------|--------|
| Yashu Gautamkumar Patel | ypatel37 |
| Vihar Ramesh Jain | vjain69 |
| Anish Pravin Kulkarni | akulka76 |
| Samir Patel | spate169 |
| Diego Miramontes | djmiramo |

## Dataset

- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — 2,484 résumés with HTML and 24 category labels. **Not included in this repository.** Place the CSV as `data/raw/UpdatedResumeDataSet.csv` after download (see setup).

## Environment

- **Python 3.11** (conda recommended).
- Core dependencies: `requirements.txt` (scikit‑learn, NLTK, spaCy, sentence‑transformers, imbalanced‑learn, mlxtend, NetworkX, etc.).

```bash
conda create -n cse-572 python=3.11 -y
conda activate cse-572
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**Optional (GPU):** `pip install torch` — used by the SentenceTransformer path (CUDA on Linux/Windows, Metal on Apple Silicon when available). Optional `pip install xgboost` enables GPU tree training where the code path supports it.

**Kaggle download example:**

```bash
pip install kaggle
kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw/ --unzip
cp data/raw/Resume/Resume.csv data/raw/UpdatedResumeDataSet.csv
```

## Running the pipeline

```bash
conda activate cse-572
python src/main.py --stage all          # full run (first time: no saved models)
python src/main.py --stage predict     # after models exist: quick evaluation
```

Staged runs: `parsing`, `features`, `mining`, `classification` (see `src/main.py`).

After the first full training, the program can load saved feature matrices and classifiers from `data/processed/` (created locally; not in git) when you choose the saved-weights option.

## Pre-trained weights (optional)

To skip training, download [pre-trained `features.pkl` and `models/trained_classifiers.pkl`](https://www.dropbox.com/scl/fo/0qtdqh2oa0ivsdro3hq1k/AIMU_1WjQov9Up4rsgdcIjY?rlkey=ulkfda1xyrpjyr01n9x0rg76w&st=8rkk1fu2&dl=0) and place them as:

- `data/processed/features.pkl`
- `data/processed/models/trained_classifiers.pkl`

Then run `python src/main.py --stage predict`. Training from scratch is still supported for reproduction.

## Results (committed sample outputs)

After a run, see `results/` for tables and figures, including:

- `classification_comparison.csv`, `confusion_matrix.png`, bar charts for accuracy and macro‑F1  
- `clustering_comparison.csv`, `tsne_*.png`  
- `association_rules_top.csv`, `cooccurrence_graph.png`  

## Tests

```bash
pytest tests/ -v
```

There are 69 unit tests; they use synthetic data and do not require the Kaggle download.

## Project layout

```
src/
  main.py
  parsing/html_parser.py
  agents/section_agents.py
  features/feature_extraction.py
  classification/classifiers.py
  clustering/cluster.py
  mining/association_rules.py
  mining/cooccurrence.py
  utils/data_loader.py
  utils/evaluation.py
data/
  raw/              # you add UpdatedResumeDataSet.csv (git-ignored)
  processed/        # features, models, parsed pickles (git-ignored)
results/            # reports and figures (selected files may be committed)
tests/
```

## Methods (course alignment)

- **Classification:** Random Forest, HistGradientBoosting, `LinearSVC` (and optional XGBoost where installed).  
- **Clustering / DR:** K‑Means, agglomerative clustering, DBSCAN; TruncatedSVD; t‑SNE plots.  
- **Mining:** FP‑Growth association rules; NetworkX co‑occurrence graph.  
- **Features:** TF‑IDF (unigrams + bigrams), χ² feature selection, section-aware and hybrid stacks, SentenceTransformer embeddings, handcrafted features (e.g. seniority, NER counts).  
- **Evaluation:** Accuracy, macro‑F1, balanced accuracy, confusion matrix; silhouette, NMI, ARI for clustering.

## ASU Research Computing (optional)

On GPU nodes, install `torch` (and `xgboost` if used), confirm `nvidia-smi`, then run `python src/main.py --stage all` from the clone directory. Use the same conda environment as on your own machine.
