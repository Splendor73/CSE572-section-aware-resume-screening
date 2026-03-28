# Section-Aware Multi-Agent Resume Screening and Skill Mining System

ASU CSE 572 — Data Mining Group Project

## Problem

Traditional resume screening relies on keyword matching, ignoring the context in which skills appear. Skills mentioned in different sections (Skills, Experience, Education) represent different expertise levels, yet most systems treat them equally. We investigate a section-aware multi-agent framework that analyzes each section independently and combines outputs for data mining — discovering career patterns, transferable skills, and improving job category prediction.

## Team

- Yashu Gautamkumar Patel (ypatel37)
- Vihar Ramesh Jain (vjain69)
- Anish Pravin Kulkarni (akulka76)
- Samir Patel (spate169)
- Diego Miramontes (djmiramo)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline
python src/main.py

# Run specific stage
python src/main.py --stage parsing
python src/main.py --stage features
python src/main.py --stage mining
python src/main.py --stage classification

# Run tests
pytest tests/
```

## Dataset

- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — 2,484 resumes with HTML structure and 24 job category labels
- Download and place in `data/raw/`
