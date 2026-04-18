"""Tests for HTML section parser."""

import pytest
import pandas as pd
from src.parsing.html_parser import parse_resume_html, parse_all_resumes


SAMPLE_HTML = """
<div class="fontsize fontface" id="document">
  <div class="section firstsection" id="SECTION_NAME123">
    <div class="name" itemprop="name">
      <span class="field" id="123LNAM1">SOFTWARE ENGINEER</span>
    </div>
  </div>
  <div class="section" id="SECTION_HILT456">
    <div class="heading">
      <div class="sectiontitle" id="SECTNAME_HILT456">Skills</div>
    </div>
    <div class="paragraph">Python, Java, SQL, machine learning, docker</div>
  </div>
  <div class="section" id="SECTION_EXPR789">
    <div class="heading">
      <div class="sectiontitle" id="SECTNAME_EXPR789">Experience</div>
    </div>
    <div class="paragraph">Software Engineer at Google, 5 years of experience building APIs</div>
  </div>
  <div class="section" id="SECTION_EDUC101">
    <div class="heading">
      <div class="sectiontitle" id="SECTNAME_EDUC101">Education</div>
    </div>
    <div class="paragraph">Bachelor of Science in Computer Science, Stanford University</div>
  </div>
</div>
"""

MINIMAL_HTML = "<div>Just some text with no sections at all</div>"


class TestParseResumeHtml:
    def test_extracts_skills(self):
        result = parse_resume_html(SAMPLE_HTML)
        assert "python" in result["skills"].lower()
        assert "java" in result["skills"].lower()

    def test_extracts_experience(self):
        result = parse_resume_html(SAMPLE_HTML)
        assert "google" in result["experience"].lower()

    def test_extracts_education(self):
        result = parse_resume_html(SAMPLE_HTML)
        assert "stanford" in result["education"].lower()
        assert "bachelor" in result["education"].lower()

    def test_returns_all_four_keys(self):
        result = parse_resume_html(SAMPLE_HTML)
        assert set(result.keys()) == {"skills", "experience", "education", "other"}

    def test_empty_input(self):
        result = parse_resume_html("")
        assert result == {"skills": "", "experience": "", "education": "", "other": ""}

    def test_none_input(self):
        result = parse_resume_html(None)
        assert result == {"skills": "", "experience": "", "education": "", "other": ""}

    def test_no_sections_fallback(self):
        result = parse_resume_html(MINIMAL_HTML)
        assert result["other"] != ""
        assert "text" in result["other"].lower()

    def test_section_id_fallback(self):
        html = """
        <div id="document">
          <div class="section" id="SECTION_SKLL100">
            <div class="paragraph">React, Node.js, TypeScript</div>
          </div>
        </div>
        """
        result = parse_resume_html(html)
        assert result["skills"] != "" or result["other"] != ""


class TestParseAllResumes:
    def test_adds_columns(self):
        df = pd.DataFrame({
            "Resume_html": [SAMPLE_HTML, MINIMAL_HTML],
            "Category": ["IT", "OTHER"],
        })
        result = parse_all_resumes(df)
        assert "skills_text" in result.columns
        assert "experience_text" in result.columns
        assert "education_text" in result.columns
        assert "other_text" in result.columns
        assert len(result) == 2

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            "Resume_html": [SAMPLE_HTML],
            "Category": ["IT"],
        })
        original_cols = list(df.columns)
        parse_all_resumes(df)
        assert list(df.columns) == original_cols
