# Usage Guide

how to actually use this thing, run predictions, and what you could extend

## quickest way to see results

if the model weights are already saved (someone trained before you):

```bash
python src/main.py --stage predict
```

this loads the saved classifiers and features, runs predictions on the test set, prints accuracy and macro-F1 for each model. takes like 2 seconds, no training needed.

if theres no saved weights itll tell you:
```
[WARNING] No saved model weights found at:
  data/processed/models/trained_classifiers.pkl

You have two options:
  1. Download the pre-trained weights and place them at:
     data/processed/models/trained_classifiers.pkl

  2. Train from scratch by running:
     python src/main.py --stage classification
```

## using on a new resume

the system is built to process the kaggle dataset as a batch but you can use pieces of it on individual resumes

### loading saved stuff

```python
import pickle

with open("data/processed/features.pkl", "rb") as f:
    features = pickle.load(f)

flat_vec = features["flat"]["vectorizer"]
section_vecs = features["section"]["vectorizers"]
label_enc = features["labels"]["label_encoder"]

# load trained classifier
with open("data/processed/models/trained_classifiers.pkl", "rb") as f:
    models = pickle.load(f)
best_clf = models["combined_HistGBT"]  # our best model
```

### processing a resume with HTML

```python
from src.parsing.html_parser import parse_resume_html
from src.agents.section_agents import SkillsAgent, ExperienceAgent, EducationAgent

sections = parse_resume_html("<div>... your html ...</div>")

skills_agent = SkillsAgent()
exp_agent = ExperienceAgent()
edu_agent = EducationAgent()

skills_tokens = skills_agent.process(sections["skills"])
exp_tokens = exp_agent.process(sections["experience"])
edu_tokens = edu_agent.process(sections["education"])

years = exp_agent.extract_years_experience(sections["experience"])
degree = edu_agent.detect_degree_level(sections["education"])
```

### plain text only (no HTML)

just use the flat vectorizer:

```python
X_new = flat_vec.transform(["paste resume text here"])
# prediction = best_clf.predict(X_new)
# category = label_enc.inverse_transform(prediction)
```

## gpu stuff

the sentence transformer encoding auto-detects the best available device:
- mac: Metal/MPS
- windows/linux with nvidia: CUDA
- nothing: CPU (slower but fine)

you dont need to configure anything. itll print what its using when you run feature extraction

## job description matching (future work)

we only do classification right now but you could extend it for resume-to-job matching:

**skill overlap approach:**
```python
from src.agents.section_agents import SkillsAgent
agent = SkillsAgent()

resume_skills = set(agent.process("Python, machine learning, SQL..."))
job_skills = set(agent.process("Required: Python, deep learning, NLP..."))
score = len(resume_skills & job_skills) / len(job_skills) if job_skills else 0
```

**cosine similarity approach:** transform both resume and job desc through same TF-IDF vectorizer and compute similarity

## things to keep in mind

1. the TF-IDF vocabulary is frozen to training data. words not in the training set get ignored
2. HTML parser was built for this specific kaggle dataset's HTML structure. different resume formats might not parse well
3. only 24 categories. for different categories you'd need to retrain on different data
4. if you retrain, old saved weights get overwritten. the predict stage uses whatever weights are on disk
