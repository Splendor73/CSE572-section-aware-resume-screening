# Section-Aware Multi-Agent Resume Screening and Skill Mining

CSE 572 Group Project -- ASU, Spring 2026

## what this is

So basically we wanted to see if breaking resumes into sections before doing feature extraction actually helps with classification. Like instead of just throwing the whole resume text into a TF-IDF vectorizer (which is what most people do), we parse out the Skills, Experience, and Education sections from the HTML and give each one to its own "agent" that knows how to handle that type of content.

Turns out the section-aware TF-IDF alone didnt beat flat TF-IDF (honestly was kinda disappointing at first). But then we added Sentence Transformer embeddings (all-MiniLM-L6-v2) on top and combined those with the flat TF-IDF features and that combo pushed us to **81% macro-F1** which is actually pretty decent for 24 classes.

We also did clustering, association rule mining, and skill co-occurrence networks because the project needed mining techniques too.

## the dataset

[Resume Dataset from Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) -- 2,484 resumes, 24 job categories (HR, IT, Finance, Chef, etc). Its kinda imbalanced, BPO only has 22 resumes while IT has 120. We used stratified 70/15/15 splits and `class_weight="balanced"` everywhere we could.

## pipeline overview

theres 4 stages that run in order:

**Stage 1 - Parsing:** BeautifulSoup pulls out section text from the HTML. Three agents process their sections (SkillsAgent normalizes synonyms like "js"->"javascript", ExperienceAgent pulls years of experience, EducationAgent detects degree level). Detection rate was like 98-100% which surprised us tbh

**Stage 2 - Features:** We build 5 different feature sets:
- Flat TF-IDF (bigrams, 8000 features -> chi2 down to 4000)
- Section-aware TF-IDF (3000 per section + handcrafted features like seniority, NER counts)
- Hybrid (flat+section+their LSA projections stacked together)
- Semantic (Sentence Transformer encodes each section separately, ~1546 dims total)
- Combined (flat TF-IDF + semantic = 5546 dims) -- **this is our best one**

The Sentence Transformer step auto-detects GPU. Uses Metal/MPS on Mac, CUDA on Windows/Linux with nvidia, falls back to CPU if nothing available.

**Stage 3 - Mining:** FP-Growth association rules on skill sets, NetworkX co-occurrence graph, K-Means/Hierarchical/DBSCAN clustering with SVD reduction

**Stage 4 - Classification:** 3 classifiers (RandomForest, HistGradientBoosting, LinearSVC) across all feature branches. Hyperparams selected on validation, then retrained on train+val. Models get saved to disk so you can skip retraining next time.

## results

best model: **HistGBT on combined features -- 83.6% accuracy, 0.810 macro-F1**

| Classifier | Feature Set | Accuracy | Macro-F1 |
|---|---|---|---|
| HistGBT | combined | 83.6% | 0.810 |
| HistGBT | flat | 79.9% | 0.782 |
| RandomForest | combined | 78.0% | 0.750 |
| RandomForest | flat | 76.9% | 0.739 |
| LinearSVC | flat | 71.3% | 0.680 |

random chance on 24 classes would be ~4.2% so yeah 81% is solid

the semantic embeddings really helped. TF-IDF alone topped out around 78% but adding the sentence transformer embeddings on top pushed it past 80. makes sense because TF-IDF is just word counts basically while the embeddings capture actual meaning

clustering silhouette scores were low (~0.05-0.10) which is expected when you have 24 categories that share tons of vocabulary (like "Business Development" and "Sales" both talk about "management" and "clients")

## how to set it up

```bash
# 1. clone and setup env
git clone https://github.com/Splendor73/CSE572-section-aware-resume-screening.git
cd CSE572-section-aware-resume-screening
conda create -n cse-572 python=3.11 -y
conda activate cse-572

# 2. install core dependencies
pip install -r requirements.txt
pip install mlxtend networkx nltk seaborn lxml imbalanced-learn
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 3. install GPU support (optional but recommended)
pip install torch xgboost
# - Mac: sentence transformer uses Metal/MPS automatically
# - Windows/Linux with NVIDIA: uses CUDA automatically
# - no GPU: falls back to CPU, still works fine
# xgboost enables GPU-accelerated classification on NVIDIA GPUs

# 4. check it worked
python -c "import sklearn, bs4, nltk, mlxtend, networkx, seaborn, sentence_transformers, spacy; print('all good')"
# check GPU (optional):
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"

# 5. get the dataset
# download from kaggle and put CSV at data/raw/UpdatedResumeDataSet.csv
# or use kaggle cli:
pip install kaggle
kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw/ --unzip
cp data/raw/Resume/Resume.csv data/raw/UpdatedResumeDataSet.csv
```

## running on ASU Sol (supercomputer)

if you have Sol access and want GPU acceleration:

```bash
# request a GPU node through OnDemand or sbatch
# partition: gpu, QOS: normal, GPUs: 1

# once on the GPU node:
cd ~/FURI/572/CSE572-section-aware-resume-screening
conda activate cse-572
pip install torch xgboost   # needed for GPU training

# verify GPU is available
nvidia-smi

# run the pipeline
python src/main.py --stage all
```

when GPU is detected, the pipeline will ask you to pick a training mode:
- **1. GPU only** -- XGBoost on GPU, fastest (~1 min), gets ~0.79 macro-F1
- **2. CPU only** -- RF + sklearn HistGBT + LinearSVC, slower (~10 min), gets best results (~0.81)
- **3. Both** -- all 4 classifiers, GPU + CPU, most comprehensive for the report

sentence transformer encoding always uses GPU automatically when available.

**if git pull says "up to date" but code looks old**, force sync:
```bash
git stash drop
git fetch origin
git reset --hard origin/main
```

## how to run it

```bash
conda activate cse-572

# full pipeline (first time, takes 5-10 min)
python src/main.py --stage all
# it will ask if you want to use saved weights or train fresh
# pick 2 on first run since theres no saved weights yet

# after that you can use saved weights for instant results
python src/main.py --stage predict

# or run individual stages
python src/main.py --stage parsing
python src/main.py --stage features
python src/main.py --stage mining
python src/main.py --stage classification
```

if you already trained once and run `--stage all` or `--stage classification` again, it will ask you:
```
What would you like to do?
  1. Use saved weights (quick, no retraining)
  2. Train again from scratch (takes 5-10 min)
```

pick 1 to just load the saved models and see results instantly. pick 2 if you changed something and need to retrain.

if theres no saved weights at all, it just trains automatically and saves them for next time.

## using pre-trained weights (best results)

we included our best-performing weights in `data/extras/`. these were trained on Windows (i7-12th gen) and achieved **0.810 macro-F1, 83.6% accuracy** on the test set with HistGBT on combined features.

download from here: [Pre-trained weights (Dropbox)](https://www.dropbox.com/scl/fo/0qtdqh2oa0ivsdro3hq1k/AIMU_1WjQov9Up4rsgdcIjY?rlkey=ulkfda1xyrpjyr01n9x0rg76w&st=8rkk1fu2&dl=0)

then place the files:

```bash
mkdir -p data/processed/models
cp <download-path>/features.pkl data/processed/features.pkl
cp <download-path>/models/trained_classifiers.pkl data/processed/models/trained_classifiers.pkl

# run predictions instantly
python src/main.py --stage predict
```

this skips all training and just loads the saved models. results will match our reported numbers exactly.

if you want to train from scratch instead, just run `--stage all` and pick option 2. results may vary slightly across different OS/hardware due to floating point differences in sklearn.

## checking the output

after running the pipeline, check `results/`:
- `classification_comparison.csv` -- all the numbers, one row per classifier x feature set
- `confusion_matrix.png` -- 24x24 heatmap for the best model
- `comparison_macro_f1.png` and `comparison_accuracy.png` -- bar charts
- `clustering_comparison.csv` -- silhouette, NMI, ARI for all clustering runs
- `tsne_*.png` -- scatter plots colored by cluster / true category
- `association_rules_top.csv` -- top 50 rules by lift
- `cooccurrence_graph.png` -- skill network viz

model weights are at `data/processed/models/trained_classifiers.pkl`

## tests

```bash
pytest tests/ -v
```

69 tests, all passing. they use synthetic data so they run in a few seconds without needing the real dataset.

## project structure

```
src/
  main.py                           # runs everything, handles weight loading
  parsing/html_parser.py            # BeautifulSoup section extraction
  agents/section_agents.py          # Skills/Experience/Education agents
  features/feature_extraction.py    # TF-IDF, semantic embeddings, chi2, etc
  classification/classifiers.py     # RF, HistGBT, LinearSVC + model saving
  clustering/cluster.py             # K-Means, Hierarchical, DBSCAN, t-SNE
  mining/association_rules.py       # FP-Growth rules
  mining/cooccurrence.py            # NetworkX skill graph
  utils/data_loader.py              # loads data, splits, save/load
  utils/evaluation.py               # metrics and plots
```

## tech stack

Python 3.11 in conda. scikit-learn, BeautifulSoup4+lxml, NLTK, sentence-transformers, spacy, mlxtend, NetworkX, matplotlib+seaborn, imbalanced-learn (SMOTE), pandas, numpy, scipy. GPU auto-detected for the transformer encoding (MPS on mac, CUDA on windows/linux).

## team

| Name | ASU ID |
|------|--------|
| Yashu Gautamkumar Patel | ypatel37 |
| Vihar Ramesh Jain | vjain69 |
| Anish Pravin Kulkarni | akulka76 |
| Samir Patel | spate169 |
| Diego Miramontes | djmiramo |
