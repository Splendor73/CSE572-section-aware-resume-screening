# CSE 572 Techniques

everything we used and how it connects to what we learned in class

## text preprocessing stuff

### TF-IDF

`src/features/feature_extraction.py`

this is like the core of our features. TF-IDF weights words by how important they are -- common words across all resumes get low weight, words unique to specific resumes get high weight. we used `sublinear_tf=True` which does a log transform so really long resumes dont dominate just because they repeat words more

we also did bigrams (`ngram_range=(1,2)`) which turned out to help. captures phrases like "machine learning" and "data analysis" that lose meaning if you split them into individual words

two versions:
- flat: one big TF-IDF on the whole resume (8000 features)
- section: separate TF-IDF per section, stacked horizontally (3000 per section)

### chi-squared feature selection

`src/features/feature_extraction.py`

after getting the TF-IDF features we use `SelectKBest` with chi2 to cut them down to the 4000 most useful ones. basically measures how correlated each feature is with the class labels. features with low chi2 are noise so we drop them. this helped a decent amount actually -- less features = less overfitting

### sentence transformer embeddings

`src/features/feature_extraction.py`

this was kind of the game changer. we use all-MiniLM-L6-v2 to get 384-dim dense embeddings for each section and the full resume. TF-IDF is just counting words but these embeddings capture actual meaning, so "software developer" and "programmer" end up close together even tho they share zero words

we encode 4 things separately (skills, experience, education, full resume) for 4x384=1536 dims, add handcrafted features, StandardScale it all. auto-detects GPU -- uses Metal on mac, CUDA on windows/linux with nvidia, cpu if nothing else

### spaCy NER

`src/features/feature_extraction.py`

we run spaCy (en_core_web_sm) to count organization names and location names in each resume. these become handcrafted features. the thought was maybe certain job categories mention more companies or locations. not a huge signal by itself but every little bit helps

### seniority detection

`src/features/feature_extraction.py`

regex that looks for keywords like "intern", "junior", "senior", "director", "VP" and maps them to 0-7. gives the classifier some info about career level that word counts cant really capture

### tokenization/lemmatization

`src/agents/section_agents.py`

standard NLTK pipeline. word_tokenize -> remove stopwords -> WordNetLemmatizer. lemmatization is less aggressive than stemming so "running" becomes "run" but "university" stays "university" (stemming would butcher it to "univers")

## classification

### Random Forest

300 trees, `class_weight="balanced"`, parallel training. we apply SMOTE before fitting to further help with class imbalance. solid performer but not our best

result: 78.0% acc, 0.750 macro-F1 (combined features)

### HistGradientBoosting (HistGBT_sklearn)

this is basically sklearn's version of XGBoost/LightGBM. builds trees sequentially where each one fixes the previous ones mistakes. 500-800 iterations, depth tuned on validation (8, 10, or 12). we actually skip SMOTE for this one -- tried it and performance dropped. gradient boosting handles imbalance on its own through the loss function

**our best classifier.** on combined features: 83.6% accuracy, 0.810 macro-F1

### XGBoost GPU (HistGBT)

same gradient boosting idea but using the XGBoost library instead of sklearn. the big advantage is it can train on NVIDIA GPUs via CUDA which makes it way faster (~1 min vs ~10 min for sklearn on CPU). results are slightly different from sklearn (0.789 vs 0.810 on combined) because the implementations differ internally. we use this for quick iteration and sklearn for final report numbers.

result: 82.6% acc, 0.789 macro-F1 (combined features, GPU)

### LinearSVC

linear SVM. fast on sparse data which is why we picked `LinearSVC` over `SVC`. `class_weight="balanced"`, high max_iter because convergence can be slow. wrapped in a `MaxAbsScaler` pipeline because LinearSVC needs features on similar scales -- without it the model would fail to converge on section/hybrid/semantic features and give garbage predictions (we learned this the hard way, it was outputting 0.005 macro-F1 before we added the scaler)

result: 71.6% acc, 0.681 macro-F1 (flat)

### SMOTE

from imbalanced-learn. creates synthetic minority class samples by interpolating between existing ones. helps RF and SVC but hurts HistGBT (we tested). we set `k_neighbors` based on the smallest class size so it doesnt crash on tiny classes like BPO (22 samples)

### model weight saving

after training, all models get saved to `data/processed/models/trained_classifiers.pkl`. next time you can load them instantly with `--stage predict` instead of retraining from scratch

## clustering

### K-Means
sweep k from 5 to 30, measure silhouette. best was around 0.10 at k=30. not great but expected for 24 overlapping categories

### Hierarchical
ward linkage with k=24 to match true categories. slightly lower silhouette than K-Means

### DBSCAN
density-based, no need to specify k. swept eps from 3-8 with min_samples=5. most points ended up as noise because theres no clear density gaps in 50-dim space

all clustering done after SVD reduction to 50 dims + StandardScaler

## dimensionality reduction

### TruncatedSVD
like PCA but works on sparse matrices. used for clustering (50 dims) and hybrid features (128 dims per branch). way more memory efficient than converting sparse to dense first

### t-SNE
just for making 2D scatter plots. we generate 4 plots: flat clusters, flat true labels, section clusters, section true labels

## mining

### FP-Growth
mines association rules over skill sets using mlxtend. one-hot encode top 200 skills, min_support=0.05. found thousands of rules. top ones make intuitive sense like microsoft office skills always appearing together (lift ~14)

### co-occurrence network
NetworkX graph. nodes = skills, edges = skills appearing in same resume. filter to 5+ co-occurrences. degree centrality shows the most connected skills

## evaluation metrics

| Metric | what it measures |
|--------|-----------------|
| Accuracy | overall % correct |
| Macro-F1 | mean of per-class F1 (our main metric) |
| Balanced Accuracy | mean of per-class recall |
| Silhouette | cluster separation quality |
| NMI | cluster-to-label agreement |
| ARI | adjusted cluster similarity |

we use macro averaging so all 24 classes count equally. without it the model could just be good at the big classes and ignore BPO/Automobile
