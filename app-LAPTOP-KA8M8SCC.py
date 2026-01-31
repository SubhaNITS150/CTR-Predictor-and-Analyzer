from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import textstat
from catboost import CatBoostRegressor
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="CTR Prediction API")
cat_model = CatBoostRegressor()
cat_model.load_model("catboost_ctr_model.cbm")

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

sbert = SentenceTransformer("all-MiniLM-L6-v2")

PERSUASION_ANCHORS = [
    "limited time offer",
    "exclusive deal just for you",
    "win big prizes",
    "best price guaranteed",
    "don’t miss this opportunity",
    "start your journey today"
]

anchor_embeddings = sbert.encode(
    PERSUASION_ANCHORS,
    convert_to_tensor=True
)

CTA_WORDS = [
    "buy", "shop", "order", "sign up", "signup", "register",
    "download", "click", "start", "get", "try", "apply",
    "claim", "win", "join", "subscribe", "book",

    "buy now", "shop now", "order now", "start now",
    "get started", "join now", "apply now", "book now",
    "reserve now", "enroll now",

    "free trial", "try free", "start free", "get free",
    "free access", "download free", "free preview",
    "no cost", "risk free", "cancel anytime",

    "save now", "get discount", "unlock savings",
    "best deal", "exclusive offer", "special price",
    "flat off", "price drop", "cashback available",
    "grab the deal",

    "limited time", "ends soon", "last chance",
    "act now", "hurry", "don’t miss",
    "today only", "offer expires", "final call",

    "learn more", "explore", "discover",
    "see how it works", "view details",
    "check it out", "find out", "watch demo",

    "get instant access", "claim your offer",
    "buy now and save", "join free today",
    "limited offer act now", "download now it’s free"
]


class AdRequest(BaseModel):
    text: str

def clean_ad_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def capital_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return min(caps / len(words), 0.6)


def sentiment_score(text: str) -> float:
    label = sentiment_pipe(text)[0]["label"]
    return {"LABEL_2": 1.0, "LABEL_1": 0.5, "LABEL_0": 0.0}[label]


def persuasion_score(text: str) -> float:
    emb = sbert.encode(text, convert_to_tensor=True)
    sim = util.cos_sim(emb, anchor_embeddings).max().item()
    return float(np.clip(sim, 0, 1))


def cta_score(text: str) -> float:
    text = text.lower()
    hits = sum(1 for w in CTA_WORDS if w in text)
    return min(hits / 3, 1.0)


def readability_score(text: str) -> float:
    score = textstat.flesch_reading_ease(text)
    return max(0, min(score / 100, 1))


@app.post("/predict")
def predict_ctr(ad: AdRequest):
    text = clean_ad_text(ad.text)

    features = np.array([[
        sentiment_score(text),
        capital_ratio(text),
        persuasion_score(text),
        cta_score(text),
        readability_score(text)
    ]])

    prediction = float(cat_model.predict(features)[0])

    return {
        "predicted_ctr": round(prediction, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1
    )

