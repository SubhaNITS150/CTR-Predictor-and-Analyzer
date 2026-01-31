import streamlit as st
import numpy as np
import textstat

from catboost import CatBoostRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CTR Prediction System",
    layout="centered"
)

st.title("📊 CTR Prediction System")
st.write(
    "Predict **Click-Through Rate (CTR)** for ad creatives using "
    "semantic and linguistic features."
)

# =====================================================
# LOAD MODEL (CACHED)
# =====================================================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_ctr_model.cbm")
    return model

model = load_model()

# =====================================================
# NLP TOOLS (PYTORCH-FREE)
# =====================================================
vader = SentimentIntensityAnalyzer()

PERSUASION_ANCHORS = [
    "limited time offer",
    "exclusive deal just for you",
    "win big prizes",
    "best price guaranteed",
    "don’t miss this opportunity",
    "start your journey today"
]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

anchor_vectors = vectorizer.fit_transform(PERSUASION_ANCHORS)

CTA_WORDS = {
    "buy", "shop", "order", "sign up", "signup", "register",
    "download", "click", "start", "get", "try", "apply",
    "claim", "win", "join", "subscribe", "book"
}

# =====================================================
# FEATURE FUNCTIONS (IDENTICAL TO FASTAPI)
# =====================================================
def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def capital_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return min(caps / len(words), 0.6)


def sentiment_score(text: str) -> float:
    return (vader.polarity_scores(text)["compound"] + 1) / 2


def persuasion_score(text: str) -> float:
    vec = vectorizer.transform([text])
    return float(cosine_similarity(vec, anchor_vectors).max())


def cta_score(text: str) -> float:
    t = text.lower()
    hits = sum(1 for w in CTA_WORDS if w in t)
    return min(hits / 3, 1.0)


def readability_score(text: str) -> float:
    try:
        score = textstat.flesch_reading_ease(text)
        return max(0.0, min(score / 100, 1.0))
    except Exception:
        return 0.5

# =====================================================
# UI – TEXT AD PREDICTION
# =====================================================
st.subheader("📝 Text Advertisement")

text_ad = st.text_area(
    "Enter ad text",
    placeholder="LIMITED TIME OFFER Buy now and save big"
)

if st.button("Predict CTR"):
    if text_ad.strip():
        text = clean_text(text_ad)

        # SAME FEATURE ORDER AS TRAINING
        features = np.array([[
            sentiment_score(text),
            capital_ratio(text),
            persuasion_score(text),
            cta_score(text),
            readability_score(text)
        ]], dtype=np.float32)

        ctr = float(model.predict(features)[0])
        ctr = np.clip(ctr, 0.0, 1.0)

        st.success(f"🎯 **Predicted CTR: {ctr * 100:.2f}%**")

        with st.expander("🔍 Feature Breakdown"):
            st.write({
                "Sentiment Score": round(sentiment_score(text), 3),
                "Capital Ratio": round(capital_ratio(text), 3),
                "Persuasion Score": round(persuasion_score(text), 3),
                "CTA Score": round(cta_score(text), 3),
                "Readability Score": round(readability_score(text), 3)
            })

    else:
        st.warning("Please enter ad text.")
