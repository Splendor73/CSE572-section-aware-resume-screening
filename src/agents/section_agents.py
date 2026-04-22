# section agents (skills / exp / education)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

synonyms = {
    "js": "javascript", "node": "nodejs", "node.js": "nodejs",
    "react.js": "reactjs", "reactjs": "reactjs", "vue.js": "vuejs",
    "angular.js": "angularjs", "express.js": "expressjs",
    "ml": "machine learning", "ai": "artificial intelligence",
    "dl": "deep learning", "nlp": "natural language processing",
    "cv": "computer vision", "aws": "amazon web services",
    "gcp": "google cloud platform", "k8s": "kubernetes",
    "docker": "docker", "tf": "tensorflow", "py": "python",
    "cpp": "cplusplus", "c++": "cplusplus", "c#": "csharp",
    "vb": "visual basic", "vb.net": "visual basic",
    "postgres": "postgresql", "mongo": "mongodb",
    "mssql": "sql server", "ms sql": "sql server",
    "mysql": "mysql", "nosql": "nosql",
    "oop": "object oriented programming",
    "ci/cd": "cicd", "ci cd": "cicd",
    "rest": "rest api", "restful": "rest api", "api": "api",
    "html5": "html", "css3": "css", "sass": "css", "scss": "css",
    "jquery": "jquery", "typescript": "typescript", "ts": "typescript",
    "golang": "go", "ms office": "microsoft office",
    "ms excel": "excel", "powerpoint": "powerpoint", "ppt": "powerpoint",
    "photoshop": "adobe photoshop", "illustrator": "adobe illustrator",
    "r programming": "r", "sas": "sas", "spss": "spss",
    "matlab": "matlab", "tableau": "tableau", "power bi": "powerbi",
    "linux": "linux", "unix": "unix", "windows": "windows",
    "macos": "macos", "ios": "ios", "android": "android",
    "swift": "swift", "kotlin": "kotlin",
    "ruby on rails": "ruby on rails", "ror": "ruby on rails",
    "django": "django", "flask": "flask",
    "spring": "spring framework", "spring boot": "spring boot",
    ".net": "dotnet", "asp.net": "dotnet",
    "scala": "scala", "hadoop": "hadoop",
    "spark": "apache spark", "pyspark": "apache spark",
    "hive": "apache hive", "kafka": "apache kafka",
    "etl": "etl", "git": "git", "svn": "subversion",
    "jira": "jira", "agile": "agile", "scrum": "scrum",
    "devops": "devops", "qa": "quality assurance", "qc": "quality control",
    "ui": "user interface", "ux": "user experience",
    "ui/ux": "ui ux design",
    "seo": "search engine optimization", "sem": "search engine marketing",
    "crm": "customer relationship management",
    "erp": "enterprise resource planning", "sap": "sap",
}
SKILL_SYNONYMS = synonyms

degree_patterns = [
    (r"\b(?:ph\.?d|doctorate|doctoral)\b", "phd"),
    (r"\b(?:master(?:\'?s)?|m\.?s\.?|m\.?a\.?|mba|m\.?eng)\b", "masters"),
    (r"\b(?:bachelor(?:\'?s)?|b\.?s\.?|b\.?a\.?|b\.?eng|b\.?tech)\b", "bachelors"),
    (r"\b(?:associate(?:\'?s)?|a\.?s\.?|a\.?a\.?)\b", "associates"),
    (r"\b(?:diploma|certificate|certification)\b", "diploma"),
    (r"\b(?:high school|ged|secondary)\b", "high_school"),
]

years_exp_patterns = [
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
    r"(?:over|more than|nearly|approximately|about)\s*(\d+)\s*(?:years?|yrs?)",
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|working)",
]


class BaseAgent:
    def process(self, text):
        if not text or not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s.+#/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return tokens


class SkillsAgent(BaseAgent):
    def process(self, text):
        toks = super().process(text)
        return [synonyms.get(t, t) for t in toks]

    def extract_skill_set(self, text):
        return set(self.process(text))


class ExperienceAgent(BaseAgent):
    def process(self, text):
        return super().process(text)

    def extract_years_experience(self, text):
        if not text: return 0
        tl = text.lower()
        found = []
        for p in years_exp_patterns:
            found += [int(m) for m in re.findall(p, tl)]
        return max(found) if found else 0


class EducationAgent(BaseAgent):
    def process(self, text):
        return super().process(text)

    def detect_degree_level(self, text):
        if not text:
            return "unknown"
        text_lower = text.lower()
        for pattern, level in degree_patterns:
            if re.search(pattern, text_lower):
                return level
        return "unknown"


def process_all_sections(df):
    skills_agent = SkillsAgent()
    exp_agent = ExperienceAgent()
    edu_agent = EducationAgent()

    df = df.copy()

    df["skills_tokens"] = df["skills_text"].apply(skills_agent.process)
    df["experience_tokens"] = df["experience_text"].apply(exp_agent.process)
    df["education_tokens"] = df["education_text"].apply(edu_agent.process)

    df["years_experience"] = df["experience_text"].apply(exp_agent.extract_years_experience)
    df["degree_level"] = df["education_text"].apply(edu_agent.detect_degree_level)
    df["num_skills"] = df["skills_tokens"].apply(len)

    print(f"\nAgent processing complete:")
    print(f"  avg skills tokens:     {df['num_skills'].mean():.1f}")
    print(f"  avg experience tokens: {df['experience_tokens'].apply(len).mean():.1f}")
    print(f"  avg education tokens:  {df['education_tokens'].apply(len).mean():.1f}")
    print(f"  years exp found in {(df['years_experience'] > 0).sum()} / {len(df)} resumes")
    for lvl, cnt in df["degree_level"].value_counts().items():
        print(f"    {lvl:<15s} {cnt}")

    return df


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset
    from src.parsing.html_parser import parse_all_resumes

    df = load_primary_dataset()
    df = parse_all_resumes(df)
    df = process_all_sections(df)

    for i in [0, 100, 500]:
        print(f"\n{'='*60}")
        print(f"Resume {i} — {df.iloc[i]['Category']}")
        print(f"  Skills tokens (first 20): {df.iloc[i]['skills_tokens'][:20]}")
        print(f"  Experience tokens (first 20): {df.iloc[i]['experience_tokens'][:20]}")
        print(f"  Education tokens (first 20): {df.iloc[i]['education_tokens'][:20]}")
        print(f"  Years experience: {df.iloc[i]['years_experience']}")
        print(f"  Degree level: {df.iloc[i]['degree_level']}")
