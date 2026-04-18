# Data Processing

how the raw data goes from CSV to feature matrices

## raw data

[Resume Dataset from Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). 2,484 resumes in a CSV:

| Column | Whats in it |
|--------|------------|
| ID | unique id |
| Resume_str | plain text version |
| Resume_html | full HTML |
| Category | job category (24 classes) |

the 24 categories: HR, Designer, IT, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

some categories are way bigger than others. IT has 120 resumes, BPO has 22. thats why we care about macro-F1 not just accuracy

## step 1: html parsing

all the resumes come from some resume builder website so the HTML is actually pretty consistent. each section is a `<div class="section">` with a `<div class="sectiontitle">` inside it

our parser (`html_parser.py`) uses BeautifulSoup + lxml to find these divs, reads the title, and matches it against keyword lists to figure out what section it is. works really well actually:
- skills found in 98.4% of resumes
- experience found in 100%
- education found in 99.3%

if a section doesnt match anything it goes in "other". if the whole HTML is weird and has no recognizable structure (rare) everything goes in "other"

## step 2: agents

each agent does the same base processing first (lowercase, remove special chars, tokenize with NLTK, remove stopwords, lemmatize) then adds its own stuff

**SkillsAgent** -- has about 100 synonym mappings. "js" -> "javascript", "c++" -> "cplusplus", "ml" -> "machine learning", etc. honestly this was kinda tedious to build but it helps because people write skills so inconsistently

**ExperienceAgent** -- regex for years of experience. catches "15+ years", "over 10 yrs in...", stuff like that. only ~22% of resumes had a clear mention, rest default to 0

**EducationAgent** -- detects highest degree. distribution we found:

| Degree | Count |
|--------|-------|
| bachelors | 1006 |
| masters | 718 |
| diploma | 305 |
| associates | 275 |
| unknown | 117 |
| high_school | 32 |
| phd | 31 |

## step 3: splitting

70/15/15 stratified split. `random_state=42` so its reproducible. gives us 1738 train, 373 val, 373 test

## step 4: feature extraction

this is where we build the actual feature matrices that go into classifiers

### flat TF-IDF
- single vectorizer on Resume_str
- `max_features=8000`, bigrams, `sublinear_tf=True`, `min_df=2`
- then chi2 feature selection keeps top 4000
- output: sparse matrix (n_samples, 4000)

### section-aware TF-IDF
- 3 separate vectorizers (skills, experience, education), 3000 features each, bigrams
- horizontally stacked with 10 handcrafted features:
  - years_experience, degree_level (0-6), num_skills
  - skills/experience/education token counts
  - total resume length (chars)
  - seniority level (0-7, regex-based)
  - org entity count, gpe entity count (from spaCy NER)
- chi2 selection down to 4000
- output: sparse matrix (n_samples, 4000)

### semantic embeddings
- Sentence Transformer (all-MiniLM-L6-v2), auto-detects GPU
- encodes skills, experience, education, full resume text separately
- 4 x 384 = 1536 dims + handcrafted features
- StandardScaled
- total ~1546 dims dense

### combined (best one)
- flat TF-IDF (4000, converted to dense) + semantic (1546) = 5546 dims
- this is what gave us 81% macro-F1

all vectorizers are fit on train only, then transform val/test. no data leakage

## saved artifacts

everything saved in `data/processed/`:

| File | Whats inside |
|------|----------|
| train/val/test.csv | the stratified splits |
| *_parsed.pkl | DataFrames with all the section columns and agent outputs |
| parsed_full.pkl | full dataset (for mining) |
| features.pkl | all feature matrices, vectorizers, scalers, label encoder |
| models/trained_classifiers.pkl | fitted classifier weights (for quick predict) |
