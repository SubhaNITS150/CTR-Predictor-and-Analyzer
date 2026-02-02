
# CTR Predictor and Analyzer

This project develops an intelligent engine for analyzing ad text with the help of natural language processing. It identifies key words, CTR-enhancing signals like CTAs, offers, urgency, and trust, and assigns weights to them based on marketing research. With the help of these variables, the engine calculates the click-through rate (CTR) in a deterministic manner without any random labeling. The engine can be used as a web service where users can analyze ad text and get immediate results.
## Objectives

- Analyze advertisement text content, images and videos.
- Extract important and impactful keywords.
- Identify CTR-boosting patterns (CTA, discounts, urgency, trust)
- Estimate CTR in a deterministic and explainable way
- Provide actionable insights to improve ad performance

## Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NLTK, spaCy, scikit-learn, TextBlob, OpenCV  
- **Environment:** Jupyter Notebook (development & analysis)  
- **Frontend:** Streamlit UI
- **Backend:** FastAPI  
- **Data Source:** Hugging Face advertisement text datasets 
- **Deployment:** Render 
## Project Structure


    .
    ├── catboost_info                   # Compiled files (alternatively `dist`)
        ├── learn/
        ├── test/
        ├── tmp/
        ├── catboost_training.json
        ├── learn_error.tsv
        ├── test_error.tsv
        └── time_left.tsv
    ├── images-dataset/         # Image ad dataset                    
        ├── p1/
        ├── p2/
    ├── ads_creative_text_sample.csv 
    ├── app.py                  # Entry point
    ├── ctr_model.cbm           # CTR Prediction Model
    ├── Keywords.csv
    ├── requirements.txt
    ├── streamlit_app.py        # Frontend
    ├── Training_Model.ipynb    # CTR Prediction training
    ├── Image_Prediction_Model.ipynb    # CTR Prediction training
    └── start.sh
    
## Demo

[https://ctr-predictor-frontend.vercel.app/]
