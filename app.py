from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import textstat

from catboost import CatBoostRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------
# App
# --------------------
app = FastAPI(title="CTR Prediction API")

# --------------------
# Load trained model
# --------------------
model = CatBoostRegressor()
model.load_model("catboost_ctr_model.cbm")
print("CatBoost model loaded")

# --------------------
# NLP tools (PyTorch-free)
# --------------------
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

# --------------------
# Request schema
# --------------------
class AdRequest(BaseModel):
    text: str


# --------------------
# Feature functions (MATCH TRAINING)
# --------------------
def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def capital_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return min(caps / len(words), 0.6)


def sentiment_score(text: str) -> float:
    # [-1, 1] → [0, 1]
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


# --------------------
# API endpoint
# --------------------
@app.post("/predict")
def predict_ctr(req: AdRequest):
    text = clean_text(req.text)

    # EXACT feature order used during training
    features = np.array([[
        sentiment_score(text),
        capital_ratio(text),
        persuasion_score(text),
        cta_score(text),
        readability_score(text)
    ]], dtype=np.float32)

    ctr = float(model.predict(features)[0])

    return {
        "predicted_ctr": round(np.clip(ctr, 0.0, 1.0), 4)
    }


# --------------------
# Windows-safe startup
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
