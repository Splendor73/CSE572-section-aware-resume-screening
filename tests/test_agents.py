"""Tests for section agents."""

import pytest
from src.agents.section_agents import (
    BaseAgent, SkillsAgent, ExperienceAgent, EducationAgent, process_all_sections
)
import pandas as pd


class TestBaseAgent:
    def setup_method(self):
        self.agent = BaseAgent()

    def test_basic_processing(self):
        tokens = self.agent.process("Python and Java programming")
        assert "python" in tokens
        assert "java" in tokens
        assert "and" not in tokens  # stopword removed

    def test_lemmatization(self):
        tokens = self.agent.process("running companies successfully")
        assert "running" in tokens or "run" in tokens
        assert "company" in tokens

    def test_empty_input(self):
        assert self.agent.process("") == []
        assert self.agent.process(None) == []

    def test_removes_short_tokens(self):
        tokens = self.agent.process("I am a good programmer")
        assert "a" not in tokens


class TestSkillsAgent:
    def setup_method(self):
        self.agent = SkillsAgent()

    def test_synonym_normalization(self):
        tokens = self.agent.process("docker and aws experience")
        assert "docker" in tokens
        assert "amazon web services" in tokens

    def test_cpp_normalization(self):
        tokens = self.agent.process("c++ programming skills")
        assert "cplusplus" in tokens

    def test_extract_skill_set(self):
        skills = self.agent.extract_skill_set("python java python sql")
        assert isinstance(skills, set)
        assert "python" in skills
        assert "java" in skills

    def test_non_synonym_passthrough(self):
        tokens = self.agent.process("tensorflow keras pandas")
        assert "tensorflow" in tokens or "keras" in tokens


class TestExperienceAgent:
    def setup_method(self):
        self.agent = ExperienceAgent()

    def test_extract_years_basic(self):
        assert self.agent.extract_years_experience("15+ years of experience") == 15

    def test_extract_years_over(self):
        assert self.agent.extract_years_experience("over 10 years in software") == 10

    def test_extract_years_multiple(self):
        text = "3 years in sales, 7 years of experience in management"
        assert self.agent.extract_years_experience(text) == 7

    def test_no_years(self):
        assert self.agent.extract_years_experience("worked at Google") == 0

    def test_empty_input(self):
        assert self.agent.extract_years_experience("") == 0
        assert self.agent.extract_years_experience(None) == 0


class TestEducationAgent:
    def setup_method(self):
        self.agent = EducationAgent()

    def test_detect_phd(self):
        assert self.agent.detect_degree_level("PhD in Computer Science") == "phd"

    def test_detect_masters(self):
        assert self.agent.detect_degree_level("Master of Science in Data") == "masters"

    def test_detect_bachelors(self):
        assert self.agent.detect_degree_level("Bachelor of Arts in English") == "bachelors"

    def test_detect_mba(self):
        assert self.agent.detect_degree_level("MBA from Harvard") == "masters"

    def test_detect_associates(self):
        assert self.agent.detect_degree_level("Associate's degree in Nursing") == "associates"

    def test_highest_degree_wins(self):
        text = "Bachelor of Science, then completed a Master of Science"
        # PhD checked first, then masters, then bachelors; masters should win
        assert self.agent.detect_degree_level(text) == "masters"

    def test_unknown(self):
        assert self.agent.detect_degree_level("studied abroad") == "unknown"

    def test_empty(self):
        assert self.agent.detect_degree_level("") == "unknown"
        assert self.agent.detect_degree_level(None) == "unknown"


class TestProcessAllSections:
    def test_adds_all_columns(self):
        df = pd.DataFrame({
            "skills_text": ["Python Java SQL"],
            "experience_text": ["5 years of experience at Google"],
            "education_text": ["Bachelor of Science from MIT"],
        })
        result = process_all_sections(df)
        assert "skills_tokens" in result.columns
        assert "experience_tokens" in result.columns
        assert "education_tokens" in result.columns
        assert "years_experience" in result.columns
        assert "degree_level" in result.columns
        assert "num_skills" in result.columns

    def test_years_extracted(self):
        df = pd.DataFrame({
            "skills_text": ["Python"],
            "experience_text": ["10 years of experience"],
            "education_text": ["BS in CS"],
        })
        result = process_all_sections(df)
        assert result.iloc[0]["years_experience"] == 10

    def test_degree_detected(self):
        df = pd.DataFrame({
            "skills_text": ["Python"],
            "experience_text": ["worked at startup"],
            "education_text": ["Master of Science in AI"],
        })
        result = process_all_sections(df)
        assert result.iloc[0]["degree_level"] == "masters"
