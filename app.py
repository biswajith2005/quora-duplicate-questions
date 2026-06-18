import streamlit as st
import pickle
import pandas as pd
from fuzzywuzzy import fuzz
from scipy.sparse import hstack, csr_matrix

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Duplicate Question Detector")
st.markdown("*Checks if two questions mean the same thing — like Quora does internally*")
st.divider()

# Input
q1 = st.text_area("Question 1", placeholder="Type your first question here...")
q2 = st.text_area("Question 2", placeholder="Type your second question here...")

def extract_features(q1, q2):
    q1_len = len(q1)
    q2_len = len(q2)
    q1_words = len(q1.split())
    q2_words = len(q2.split())
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    common = len(w1 & w2)
    total = q1_words + q2_words
    share = round(common / total, 2) if total > 0 else 0
    fuzzy = fuzz.ratio(q1, q2)
    fuzz_partial = fuzz.partial_ratio(q1, q2)
    fuzz_token_sort = fuzz.token_sort_ratio(q1, q2)
    fuzz_token_set = fuzz.token_set_ratio(q1, q2)

    hand = csr_matrix([[q1_len, q2_len, q1_words, q2_words,
                        common, share, fuzzy,
                        fuzz_partial, fuzz_token_sort, fuzz_token_set]])

    combined_text = q1 + ' ' + q2
    tfidf_features = tfidf.transform([combined_text])

    return hstack([tfidf_features, hand])

if st.button("Check", use_container_width=True):
    if q1.strip() == '' or q2.strip() == '':
        st.warning("Please enter both questions")
    else:
        features = extract_features(q1, q2)
        result = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        st.divider()
        if result == 1:
            st.success("✅ These questions ARE duplicates")
            st.metric("Confidence", f"{round(prob[1] * 100, 1)}%")
        else:
            st.error("❌ These questions are NOT duplicates")
            st.metric("Confidence", f"{round(prob[0] * 100, 1)}%")

        with st.expander("See feature breakdown"):
            st.write(f"Common words: {len(set(q1.lower().split()) & set(q2.lower().split()))}")
            st.write(f"Fuzzy match score: {fuzz.ratio(q1, q2)}%")
            st.write(f"Token sort ratio: {fuzz.token_sort_ratio(q1, q2)}%")

st.divider()
st.caption(" NLP Project | Accuracy: 79.03%")