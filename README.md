# 🔍 Quora Duplicate Question Detector

An NLP project that detects whether two questions are semantically 
identical — replicating the core problem Quora solves at scale.

## Problem Statement
Quora receives thousands of questions daily. Many are duplicates 
of existing questions. This model automatically flags them before 
they go live — reducing content moderation workload.

## Accuracy Progression
| Attempt | Features Used | Accuracy |
|---|---|---|
| Attempt 1 | Basic features only | 72.54% |
| Attempt 2 | + Fuzzy matching features | 73.64% |
| Attempt 3 | + TF-IDF (3000 features) | 79.03% |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- NLTK, FuzzyWuzzy
- TF-IDF Vectorization
- Streamlit

## Key Features Engineered
- Question length difference
- Common word count and share ratio
- Fuzzy matching scores (ratio, partial, token sort, token set)
- TF-IDF on combined question text (3000 features)

## Model Limitations & Next Steps
- TF-IDF fails on synonym-based duplicates
- False positives when questions share common words
- Next step: Replace TF-IDF with BERT sentence transformers

## Live Demo
[Click here to try it](https://biswa-duplicate-detector.streamlit.app/)

## How to Run
pip install -r requirements.txt
streamlit run app.py

## What I Learned
- End to end NLP pipeline
- Feature engineering for text data
- Why F1 score matters more than accuracy on imbalanced data
- Iterative model improvement approach