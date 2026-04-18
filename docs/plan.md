# Implementation Plan

## what we built (all done)

- [x] Step 0: env setup, dataset download (conda cse-572, python 3.11)
- [x] Step 1: HTML section parser (BeautifulSoup, 98-100% detection)
- [x] Step 2: section agents (Skills/Experience/Education, synonym normalization, regex extraction)
- [x] Step 3: feature extraction (flat TF-IDF, section TF-IDF, hybrid LSA, semantic embeddings, combined)
- [x] Step 4: classification (RF, HistGBT, LinearSVC with validation tuning and SMOTE)
- [x] Step 5: clustering (K-Means, Hierarchical, DBSCAN with SVD reduction + t-SNE)
- [x] Step 6: mining (FP-Growth rules, NetworkX co-occurrence graph)
- [x] Step 7: pipeline orchestrator with stage CLI
- [x] Step 8: model weight saving + predict mode
- [x] Step 9: GPU auto-detection for transformer encoding
- [x] Step 10: docs and README
- [x] Step 11: XGBoost GPU training for classification (CUDA)
- [x] Step 12: training mode prompt (GPU only / CPU only / Both)
- [x] Step 13: feature caching with user prompt
- [x] Step 14: LinearSVC MaxAbsScaler fix for convergence
- [x] Step 15: pre-trained weights on Dropbox

## performance timeline

started at 0.639 macro-F1 with basic flat TF-IDF + Random Forest. got to 0.810 through:

1. added bigrams to TF-IDF -> ~0.70
2. chi2 feature selection (8000->4000) -> ~0.72
3. SMOTE for RF/SVC (skip for HistGBT) -> ~0.74
4. added HistGradientBoosting classifier -> ~0.78
5. sentence transformer embeddings + combined features -> **0.810**
6. XGBoost GPU training -> 0.789 (faster but slightly lower)
7. expanded HistGBT tuning grid (depth 8-12, lr 0.03-0.1, 500-800 iters)

key insight: SMOTE actually hurt HistGBT (0.74 with vs 0.78 without). gradient boosting handles imbalance on its own

also: results vary across OS. Windows (Intel MKL) got 0.810, Mac (Accelerate) got 0.780 with same code. floating point differences in sklearn

## whats in the repo

```
src/
  main.py                          # runs everything, weight loading, train-or-predict prompt
  parsing/html_parser.py           # BeautifulSoup section extraction
  agents/section_agents.py         # 3 agents with synonym/regex processing
  features/feature_extraction.py   # all feature branches, GPU auto-detect
  classification/classifiers.py    # 3 classifiers, model saving
  clustering/cluster.py            # 3 clustering methods + t-SNE
  mining/association_rules.py      # FP-Growth
  mining/cooccurrence.py           # NetworkX graph
  utils/data_loader.py             # data loading/splitting
  utils/evaluation.py              # metrics/plots
tests/                             # 69 unit tests, all passing
```

## techniques used (CSE 572 coverage)

- classification: Random Forest, Gradient Boosting, SVM
- clustering: K-Means, Hierarchical, DBSCAN
- dimensionality reduction: TruncatedSVD, t-SNE
- feature engineering: TF-IDF (unigrams+bigrams), chi2 selection, Sentence Transformer embeddings, NER features, handcrafted features
- preprocessing: tokenization, lemmatization, stopword removal, SMOTE
- evaluation: accuracy, macro-F1, balanced accuracy, confusion matrix, silhouette, NMI, ARI
- association rules: FP-Growth
- network analysis: co-occurrence graph with degree centrality
