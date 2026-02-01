from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import cv2
import pytesseract
import textstat
import tempfile
import os

from catboost import CatBoostRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi.middleware.cors import CORSMiddleware

# --------------------
# App
# --------------------
app = FastAPI(title="Multimodal CTR Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------
# Load model
# --------------------
model = CatBoostRegressor()
model.load_model("catboost_ctr_model_v2.cbm")
print("✅ CatBoost multimodal model loaded")

# --------------------
# OCR setup (Windows)
# --------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------
# NLP tools
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

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
anchor_vectors = vectorizer.fit_transform(PERSUASION_ANCHORS)

CTA_WORDS = {
    "buy", "shop", "order", "sign up", "signup", "register",
    "download", "click", "start", "get", "try", "apply",
    "claim", "win", "join", "subscribe", "book"
}

# --------------------
# Schemas
# --------------------
class TextAdRequest(BaseModel):
    text: str


# --------------------
# Feature helpers
# --------------------
def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def sentiment_score(text: str) -> float:
    return (vader.polarity_scores(text)["compound"] + 1) / 2


def capital_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return min(caps / len(words), 0.6)


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


def extract_image_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean() / 255
    contrast = gray.std() / 128
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255

    face_present = 0

    return [
        np.clip(brightness, 0, 1),
        np.clip(contrast, 0, 1),
        np.clip(sharpness, 0, 1),
        np.clip(edge_density, 0, 1),
        face_present
    ]


def extract_ocr_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    return clean_text(text)


# ======================================================
# 📝 TEXT AD ENDPOINT
# ======================================================
@app.post("/predict-text")
def predict_text_ad(req: TextAdRequest):

    text = clean_text(req.text)

    features = np.array([[
        sentiment_score(text),
        capital_ratio(text),
        persuasion_score(text),
        cta_score(text),
        readability_score(text),

        # Image defaults
        0.5, 0.5, 0.5, 0.5, 0
    ]], dtype=np.float32)

    ctr = float(model.predict(features)[0])

    return {
        "input_type": "text",
        "predicted_ctr": round(np.clip(ctr, 0.0, 1.0), 4)
    }


# ======================================================
# 🖼️ IMAGE AD ENDPOINT
# ======================================================
@app.post("/predict-image")
async def predict_image_ad(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        img_path = tmp.name

    img = cv2.imread(img_path)
    os.remove(img_path)

    if img is None:
        return {"error": "Invalid image"}

    ocr_text = extract_ocr_text(img)

    text_feats = [
        sentiment_score(ocr_text),
        capital_ratio(ocr_text),
        persuasion_score(ocr_text),
        cta_score(ocr_text),
        readability_score(ocr_text)
    ]

    img_feats = extract_image_features(img)

    features = np.array([text_feats + img_feats], dtype=np.float32)

    ctr = float(model.predict(features)[0])

    return {
        "input_type": "image",
        "ocr_text": ocr_text,
        "predicted_ctr": round(np.clip(ctr, 0.0, 1.0), 4)
    }


# --------------------
# Local run
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
