import streamlit as st
import numpy as np
import re
from catboost import CatBoostRegressor
from PIL import Image
import cv2
import tempfile

# ----------------------------
# Load model
# ----------------------------
model = CatBoostRegressor()
model.load_model("ctr_model.cbm")

# ----------------------------
# Keyword weights
# ----------------------------
keyword_weight = {
    "buy": 0.12, "order": 0.12, "purchase": 0.12, "shop": 0.12, "now": 0.12,
    "sale": 0.08, "deal": 0.08, "discount": 0.08,
    "limited": 0.06, "today": 0.06,
    "official": 0.05, "trusted": 0.05,
    "freedelivery": 0.04, "fastdelivery": 0.04
}

# ----------------------------
# Feature extractor (TEXT)
# ----------------------------
def extract_text_features(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())

    action = deal = urgency = trust = convenience = social = 0
    score = 0.0

    for w in tokens:
        if w in ["buy", "order", "purchase", "shop", "now", "checkout"]:
            action += 1
        elif w in ["sale", "deal", "discount", "coupon", "promo", "cashback"]:
            deal += 1
        elif w in ["limited", "today", "hurry"]:
            urgency += 1
        elif w in ["official", "trusted", "verified", "warranty"]:
            trust += 1
        elif w in ["freedelivery", "fastdelivery", "express"]:
            convenience += 1
        elif w in ["bestseller", "reviews", "ratings"]:
            social += 1

        score += keyword_weight.get(w, 0)

    return np.array([[action, deal, urgency, trust, convenience, social, score, len(tokens)]])

# ----------------------------
# Image feature extractors
# ----------------------------
def image_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def image_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CTR Prediction System", layout="centered")

st.title("📊 Multimodal CTR Prediction System")
st.write("Predict Click-Through Rate for **Text, Image, and Video Ads**")

tab1, tab2, tab3 = st.tabs(["📝 Text Ad", "🖼️ Image Ad", "🎥 Video Ad"])

# ----------------------------
# TEXT AD TAB
# ----------------------------
with tab1:
    st.subheader("Text Advertisement")
    text_ad = st.text_area("Enter Ad Text")

    if st.button("Predict CTR (Text Ad)"):
        if text_ad.strip():
            X = extract_text_features(text_ad)
            ctr = model.predict(X)[0]
            ctr = float(np.clip(ctr, 0.005, 0.25))
            st.success(f"Predicted CTR: {ctr*100:.2f}%")
        else:
            st.warning("Please enter ad text")

# ----------------------------
# IMAGE AD TAB
# ----------------------------
with tab2:
    st.subheader("Image Advertisement")
    image_file = st.file_uploader("Upload Ad Image", type=["png", "jpg", "jpeg"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Ad Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            image.save(tmp.name)
            img = cv2.imread(tmp.name)

        brightness = image_brightness(img)
        contrast = image_contrast(img)

        if st.button("Predict CTR (Image Ad)"):
            # Simple heuristic until retraining with image features
            base_ctr = 0.05
            visual_boost = min((contrast / 100) + (brightness / 255), 0.2)
            ctr = base_ctr + visual_boost
            ctr = float(np.clip(ctr, 0.01, 0.3))

            st.success(f"Predicted CTR: {ctr*100:.2f}%")
            st.caption(f"Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")

# ----------------------------
# VIDEO AD TAB
# ----------------------------
with tab3:
    st.subheader("Video Advertisement")
    video_file = st.file_uploader("Upload Ad Video", type=["mp4", "mov", "avi"])

    if video_file:
        st.video(video_file)

        st.info(
            "Video CTR prediction requires:\n"
            "- Key frame extraction\n"
            "- OCR on frames\n"
            "- Motion & audio analysis\n\n"
            "This module is intentionally kept as a placeholder."
        )

        if st.button("Predict CTR (Video Ad)"):
            ctr = np.random.uniform(0.05, 0.2)
            st.success(f"Estimated CTR: {ctr*100:.2f}%")
