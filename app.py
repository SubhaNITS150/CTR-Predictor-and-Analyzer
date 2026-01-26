from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import re
import numpy as np

app = FastAPI()

# Load model once (VERY IMPORTANT)
model = CatBoostRegressor()
model.load_model("ctr_model.cbm")
print("Model loaded successfully")


# keyword weights (same as training)
keyword_weight = {
    "buy": 0.12, "order": 0.12, "purchase": 0.12, "shop": 0.12, "now": 0.12,
    "sale": 0.08, "deal": 0.08, "discount": 0.08,
    "limited": 0.06, "today": 0.06,
    "official": 0.05, "trusted": 0.05,
    "freedelivery": 0.04, "fastdelivery": 0.04
}

# feature extractor (MUST MATCH TRAINING)
def extract_features(ad_text: str):
    tokens = re.findall(r"[a-zA-Z]+", ad_text.lower())

    action_cnt = 0
    deal_cnt = 0
    urgency_cnt = 0
    trust_cnt = 0
    convenience_cnt = 0
    socialproof_cnt = 0
    total_keyword_score = 0.0

    for word in tokens:
        if word in ["buy", "order", "purchase", "shop", "now", "checkout"]:
            action_cnt += 1
        elif word in ["sale", "deal", "discount", "coupon", "promo", "cashback", "clearance"]:
            deal_cnt += 1
        elif word in ["limited", "hurry", "lastchance", "endingsoon", "flashsale", "today"]:
            urgency_cnt += 1
        elif word in ["original", "genuine", "official", "trusted", "verified", "warranty"]:
            trust_cnt += 1
        elif word in ["freedelivery", "freeshipping", "fastdelivery", "instant", "express"]:
            convenience_cnt += 1
        elif word in ["bestseller", "toprated", "reviews", "ratings", "recommended"]:
            socialproof_cnt += 1

        total_keyword_score += keyword_weight.get(word, 0)

    ad_length = len(tokens)

    # IMPORTANT: feature order MUST match training
    return np.array([[
        action_cnt,
        deal_cnt,
        urgency_cnt,
        trust_cnt,
        convenience_cnt,
        socialproof_cnt,
        total_keyword_score,
        ad_length
    ]])



class AdRequest(BaseModel):
    ad_text: str

@app.post("/predict")
def predict_ctr(req: AdRequest):
    X = extract_features(req.ad_text)

    ctr = model.predict(X)[0]
    ctr = float(np.clip(ctr, 0.005, 0.25))

    return {
        "ad_text": req.ad_text,
        "predicted_ctr": round(ctr, 4),
        "predicted_ctr_percent": round(ctr * 100, 2)
    }

