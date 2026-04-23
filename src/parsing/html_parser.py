# html -> skills / experience / education text

from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import Dict, List, Any

# keywords for matching section headers
section_keywords = {
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "expertise", "highlights", "qualifications summary", "areas of expertise",
        "technical proficiencies", "proficiencies", "technologies",
    ],
    "experience": [
        "experience", "work experience", "employment", "work history",
        "professional experience", "employment history", "professional background",
        "accomplishments", "professional summary", "summary",
    ],
    "education": [
        "education", "academic", "qualifications", "certifications",
        "certification", "training", "academic background", "licenses",
        "education and training", "credentials",
    ],
}

SECTION_KEYWORDS = section_keywords


def _classify_section(title_text: str) -> str:
    """Classify a given section title into a predefined category based on keywords."""
    title_lower = title_text.strip().lower()
    title_clean = re.sub(r"[^a-z\s]", "", title_lower).strip()
    for section, kws in section_keywords.items():
        for kw in kws:
            if kw == title_clean or title_clean.startswith(kw) or kw in title_clean:
                return section
    return "other"


def _get_section_text(section_div: Any) -> str:
    """Extract and clean text content from a BeautifulSoup section div."""
    txt = section_div.get_text(separator=" ", strip=True)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def parse_resume_html(html_str: str) -> Dict[str, str]:
    """
    Parse a resume HTML string into categorized sections (skills, experience, education).
    
    Args:
        html_str (str): The raw HTML string of the resume.
        
    Returns:
        Dict[str, str]: A dictionary mapping section categories to their text content.
    """
    result = {"skills": "", "experience": "", "education": "", "other": ""}

    if not html_str or not isinstance(html_str, str):
        return result

    soup = BeautifulSoup(html_str, "lxml")
    sections = soup.find_all("div", class_="section")

    if not sections:
        result["other"] = _get_section_text(soup)
        return result

    for sec in sections:
        title_div = sec.find("div", class_="sectiontitle")

        if title_div:
            title_text = title_div.get_text(strip=True)
            cat = _classify_section(title_text)
        else:
            sec_id = sec.get("id", "")
            if "NAME" in sec_id: cat = "other"
            elif "EXPR" in sec_id or "SUMM" in sec_id: cat = "experience"
            elif "HILT" in sec_id or "SKLL" in sec_id: cat = "skills"
            elif "EDUC" in sec_id or "CERT" in sec_id: cat = "education"
            else: cat = "other"

        text = _get_section_text(sec)
        if text:
            if title_div:
                title_text_clean = title_div.get_text(strip=True)
                if text.startswith(title_text_clean):
                    text = text[len(title_text_clean):].strip()

            if result[cat]:
                result[cat] += " " + text
            else:
                result[cat] = text

    return result


def parse_all_resumes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply HTML parsing across the entire dataset to extract structured sections.
    
    Args:
        df (pd.DataFrame): The input dataframe containing a 'Resume_html' column.
        
    Returns:
        pd.DataFrame: A new dataframe augmented with parsed section text columns.
    """
    parsed = df["Resume_html"].apply(parse_resume_html)

    df = df.copy()
    df["skills_text"] = parsed.apply(lambda x: x["skills"])
    df["experience_text"] = parsed.apply(lambda x: x["experience"])
    df["education_text"] = parsed.apply(lambda x: x["education"])
    df["other_text"] = parsed.apply(lambda x: x["other"])

    total = len(df)
    has_skills = (df["skills_text"].str.len() > 0).sum()
    has_exp = (df["experience_text"].str.len() > 0).sum()
    has_edu = (df["education_text"].str.len() > 0).sum()
    has_any = ((df["skills_text"].str.len() > 0) |
               (df["experience_text"].str.len() > 0) |
               (df["education_text"].str.len() > 0)).sum()

    print(f"\nSection extraction results ({total} resumes):")
    print(f"  Skills detected:     {has_skills:>5d} ({has_skills/total*100:.1f}%)")
    print(f"  Experience detected: {has_exp:>5d} ({has_exp/total*100:.1f}%)")
    print(f"  Education detected:  {has_edu:>5d} ({has_edu/total*100:.1f}%)")
    print(f"  Any section found:   {has_any:>5d} ({has_any/total*100:.1f}%)")

    return df


if __name__ == "__main__":
    from src.utils.data_loader import load_primary_dataset

    df = load_primary_dataset()
    df = parse_all_resumes(df)

    for i in [0, 100, 500]:
        print(f"\n{'='*60}")
        print(f"Resume {i} — Category: {df.iloc[i]['Category']}")
        for col in ["skills_text", "experience_text", "education_text"]:
            text = df.iloc[i][col]
            print(f"  {col}: {text[:150]}..." if len(text) > 150 else f"  {col}: {text}")
