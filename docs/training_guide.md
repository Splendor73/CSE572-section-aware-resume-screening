# Training Guide

how to train, whats going on under the hood, and how to just use saved weights without retraining

## quick start

if you just want results and dont want to wait:

```bash
# if weights already exist (someone already trained)
python src/main.py --stage predict

# if no weights, train everything from scratch
python src/main.py --stage all
# when it asks, pick 2 to train fresh
```

or download our best pre-trained weights from [Dropbox](https://www.dropbox.com/scl/fo/0qtdqh2oa0ivsdro3hq1k/AIMU_1WjQov9Up4rsgdcIjY?rlkey=ulkfda1xyrpjyr01n9x0rg76w&st=8rkk1fu2&dl=0), put them at `data/processed/features.pkl` and `data/processed/models/trained_classifiers.pkl`, then run `--stage predict`.

after training once, weights are saved at `data/processed/models/trained_classifiers.pkl`. next time you can just load those.

## GPU stuff

### sentence transformer encoding
auto-detects your GPU:
- **Mac:** uses Metal/MPS
- **Windows/Linux with nvidia:** uses CUDA
- **no GPU:** falls back to CPU (slower but works fine)

you dont need to configure anything, it just picks the best available option. with GPU the encoding takes maybe 30 seconds, on CPU its more like 2-3 minutes.

### classification training
when a CUDA GPU is detected, the pipeline asks what training mode you want:

```
  Training mode options:

    1. GPU only (recommended for quick runs)
       - Trains XGBoost HistGBT on GPU (~1 min total)
       - Only 1 classifier but its our best architecture
       - Expected: ~0.78-0.79 macro-F1 on combined features

    2. CPU only (recommended for final report numbers)
       - Trains RandomForest + sklearn HistGBT + LinearSVC
       - sklearn HistGBT historically gives best results (~0.81)
       - Takes ~10-15 min, all on CPU

    3. Both GPU + CPU (most comprehensive)
       - Runs all 4 classifiers: XGBoost(GPU) + RF + HistGBT + SVC(CPU)
       - Best for comparing GPU vs CPU performance in the report
       - Takes ~10-15 min (CPU classifiers are the bottleneck)
```

GPU only uses XGBoost which trains on NVIDIA GPUs via CUDA. sklearn classifiers (RF, HistGBT, LinearSVC) are CPU-only, theres no way around that.

on Mac (MPS) or machines without CUDA, it just runs all 3 CPU classifiers automatically, no prompt.

## classifier configs

| Classifier | Settings | Notes |
|---|---|---|
| RandomForest | 300 or 500 trees, balanced weights | uses SMOTE, parallel with n_jobs=-1 |
| HistGBT_sklearn | 500-800 iters, depth 8-12, lr 0.03-0.1 | no SMOTE (it does worse with it), tuned on val |
| HistGBT (XGBoost) | 500 trees, depth 8 or 10 | GPU-accelerated via CUDA, no SMOTE |
| LinearSVC | max_iter=50000, balanced weights | wrapped in MaxAbsScaler pipeline for convergence |

### about SMOTE

we use SMOTE (synthetic oversampling) for RF and LinearSVC to balance the classes. but for HistGBT we skip it -- we tested and it actually hurt performance. gradient boosting handles imbalance internally through its loss function. took us a while to figure that out, HistGBT was doing 0.74 with SMOTE and jumped to 0.78 without it

### about LinearSVC scaling

LinearSVC needs features on similar scales to converge. without scaling it would fail on section/hybrid/semantic features (different scales from TF-IDF vs embeddings vs handcrafted). we wrap it in a `MaxAbsScaler` pipeline so it scales without destroying sparsity.

### hyperparameter tuning

small grid search on validation set, pick best by macro-F1, then retrain on train+val combined before final test evaluation. grids are intentionally small to avoid overfitting to val:

- RF: 300 vs 500 trees
- HistGBT sklearn: depth 8/10/12, lr 0.1/0.05/0.03, iters 500/800
- HistGBT XGBoost: depth 8 vs 10, lr 0.1 vs 0.05
- LinearSVC: C = 0.5, 1.0, or 2.0

## running different stages

```bash
# full thing (parsing -> features -> mining -> classification)
python src/main.py --stage all

# just classification (loads features.pkl, trains classifiers)
python src/main.py --stage classification

# just load saved weights and evaluate (instant)
python src/main.py --stage predict

# single module for debugging
python -m src.classification.classifiers
```

when you run `--stage all`, the pipeline asks at each major step:

1. **features step:** if `features.pkl` exists, asks if you want to reuse or re-extract
2. **classification step:** if saved weights exist, asks if you want to load or retrain
3. **training mode:** if GPU detected, asks GPU only / CPU only / both

if theres no cached files it just runs everything automatically.

## what we measure

| Metric | Why |
|---|---|
| Macro-F1 | **main metric.** mean of per-class F1 so all 24 categories count equally |
| Accuracy | easy to understand but misleading with imbalanced classes |
| Balanced Accuracy | mean of per-class recall, good sanity check |
| Macro Precision | per-class correctness |
| Macro Recall | per-class coverage |

## latest results

best model: **HistGBT_sklearn on combined features -- 83.6% accuracy, 0.810 macro-F1** (trained on Windows i7-12th gen)

| Classifier | Feature Set | Accuracy | Macro-F1 |
|---|---|---|---|
| HistGBT_sklearn | combined | 83.6% | 0.810 |
| HistGBT_sklearn | flat | 79.6% | 0.772 |
| HistGBT_sklearn | hybrid | 78.6% | 0.753 |
| RandomForest | flat | 78.8% | 0.750 |
| HistGBT (XGBoost GPU) | combined | 82.6% | 0.789 |
| LinearSVC | flat | 71.6% | 0.681 |

the combined feature set (flat TF-IDF + sentence transformer embeddings) is clearly the winner. semantic embeddings alone do about 0.63 which is meh, but when you combine them with TF-IDF the model gets both lexical and semantic signal and that pushes it past 0.80

note: results vary slightly across OS/hardware. sklearn's floating point math differs between Windows (Intel MKL) and Mac (Accelerate). our best 0.810 was on Windows.

clustering silhouettes are low (~0.05-0.10) but thats expected for 24 categories with tons of vocab overlap

## output files

everything goes to `results/`:

| File | What |
|------|------|
| `classification_comparison.csv` | all numbers, one row per classifier x feature set |
| `confusion_matrix.png` | heatmap for best model |
| `comparison_macro_f1.png` | bar chart |
| `comparison_accuracy.png` | bar chart |
| `clustering_comparison.csv` | clustering metrics |

model weights: `data/processed/models/trained_classifiers.pkl`
pre-trained weights: [Dropbox link](https://www.dropbox.com/scl/fo/0qtdqh2oa0ivsdro3hq1k/AIMU_1WjQov9Up4rsgdcIjY?rlkey=ulkfda1xyrpjyr01n9x0rg76w&st=8rkk1fu2&dl=0)

## changing things

**add a classifier:** add it to `get_classifiers()` in `classifiers.py`. it auto-includes in the comparison

**change TF-IDF:** edit `build_flat_tfidf()` or `build_section_tfidf()` in `feature_extraction.py`. key params: `max_features`, `ngram_range`, `min_df`. rerun from features stage after

**change the split:** edit `split_dataset()` in `data_loader.py`. default 70/15/15 random_state=42. rerun from parsing stage
