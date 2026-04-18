# System Architecture

## how it all fits together

ok so the basic idea is pretty simple. instead of treating a resume as one blob of text we break it apart into sections and process each one separately. we have three "agents" (theyre just python classes but we call them agents because it sounds cooler and also thats what the project proposal said lol) that each handle one section type.

then we extract features from those sections in multiple ways -- TF-IDF, semantic embeddings, handcrafted stuff -- and feed everything into classifiers.

the whole pipeline looks like this:

```
                        +------------------+
                        | Raw HTML Resumes |
                        +--------+---------+
                                 |
                      Stage 1: Parsing
                                 |
                   +-------------+-------------+
                   |             |             |
            Skills Text   Experience Text  Education Text
                   |             |             |
            +------+------+  +--+--+  +-------+-------+
            | SkillsAgent |  | Exp |  | EducationAgent|
            |  - synonyms |  | Agt |  |  - degree     |
            |  - normalize|  | -yr |  |    detection  |
            +------+------+  +--+--+  +-------+-------+
                   |             |             |
                   +------+------+------+------+
                          |
              Stage 2: Feature Extraction
              /      |       |       |      \
         Flat     Section  Hybrid  Semantic  Combined
        TF-IDF    TF-IDF   (LSA)   (SBERT)  (best!)
        (4000)    (4000+)          (1546)   (5546)
                          |
          +---------------+----------------+----------+
          |                   |                    |
   Stage 3: Mining     Stage 3: Clustering   Stage 4: Classification
   - FP-Growth rules   - K-Means            - RF, HistGBT_sklearn (CPU)
   - Co-occurrence     - Hierarchical        - XGBoost HistGBT (GPU)
     network           - DBSCAN              - LinearSVC (CPU)
                                             - model weights saved
```

## the agents

all three inherit from `BaseAgent` which does the standard NLP pipeline -- lowercase, remove junk chars, tokenize with NLTK, remove stopwords, lemmatize. then each adds its own thing:

**SkillsAgent** -- has a dictionary of ~100 synonym mappings. so "js" becomes "javascript", "c++" becomes "cplusplus" (bc the plus signs mess up tokenization), "ml" expands to "machine learning". this helps because people write skills so many different ways

**ExperienceAgent** -- runs regex to pull out years of experience. catches stuff like "15+ years of experience" or "over 10 years in finance". takes the max if it finds multiple. only about 22% of resumes had a clear years-of-experience mention though

**EducationAgent** -- detects highest degree mentioned (PhD, Masters, Bachelors, etc). maps to ordinal scale 0-6. checks in order from highest to lowest so if you mention both BS and MS it returns Masters

## feature branches

this is where it gets interesting. we tried a bunch of different feature representations:

**flat TF-IDF** -- the baseline. just one vectorizer on the raw resume text, bigrams, 8000 features then chi2 to keep the best 4000. simple but actually pretty solid

**section-aware TF-IDF** -- separate vectorizer per section (3000 each) stacked together with 10 handcrafted features (years exp, degree, token counts, resume length, seniority level from regex, ORG/GPE entity counts from spaCy). chi2 selected down to 4000

**hybrid** -- takes flat + section and adds their TruncatedSVD projections (128 dims each). idea was to capture latent topics. helped a little but wasnt the big win

**semantic** -- this is the Sentence Transformer stuff. we encode each section and the full resume with all-MiniLM-L6-v2 (384 dims each x 4 = 1536) plus handcrafted features, all StandardScaled. ~1546 dims total. auto-detects GPU (MPS on mac, CUDA on nvidia)

**combined** -- flat TF-IDF (4000 dense) + semantic (1546) = 5546 dims. **this is our best feature set** because it gets both the lexical signal from TF-IDF and the semantic signal from the transformer

## GPU acceleration

two things use GPU when available:

1. **sentence transformer encoding** -- auto-detects MPS (mac) or CUDA (nvidia). falls back to CPU if nothing found
2. **XGBoost classification** -- when CUDA GPU is detected, the pipeline asks if you want GPU-only training (XGBoost, ~1 min), CPU-only (sklearn RF+HistGBT+SVC, ~10 min), or both. sklearn classifiers dont have GPU support so thats the bottleneck

our best result (0.810 macro-F1) came from sklearn HistGBT on CPU. XGBoost GPU gets ~0.789 -- slightly lower because the implementations differ internally. we keep both options because GPU is way faster for iteration but CPU gives the best final numbers

## model weight saving

after training, all fitted classifiers get saved to `data/processed/models/trained_classifiers.pkl`. next time you run the pipeline it asks if you want to reuse those weights or train fresh. you can also do `--stage predict` to just load weights and evaluate instantly without any training at all.

we also have pre-trained weights on [Dropbox](https://www.dropbox.com/scl/fo/0qtdqh2oa0ivsdro3hq1k/AIMU_1WjQov9Up4rsgdcIjY?rlkey=ulkfda1xyrpjyr01n9x0rg76w&st=8rkk1fu2&dl=0) that give the 0.810 result right out of the box

## files

| File | What it does |
|------|-------------|
| `src/parsing/html_parser.py` | parses HTML into section text |
| `src/agents/section_agents.py` | 3 agents for section-specific processing |
| `src/features/feature_extraction.py` | builds all 5 feature branches, auto GPU for transformer |
| `src/classification/classifiers.py` | trains classifiers, saves/loads model weights |
| `src/clustering/cluster.py` | K-Means, Hierarchical, DBSCAN + t-SNE plots |
| `src/mining/association_rules.py` | FP-Growth rules |
| `src/mining/cooccurrence.py` | NetworkX skill graph |
| `src/utils/data_loader.py` | loads CSV, does stratified splits |
| `src/utils/evaluation.py` | metrics + plotting |
| `src/main.py` | orchestrates everything, handles train vs predict |
