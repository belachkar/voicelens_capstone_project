# TODO: Import your package, replace this by explicit imports of what you need
import tempfile

import joblib
import pandas as pd
import requests
import spacy
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from voicelens.cleaning.preprocessing import apply_cleaning
# from voicelens.main import predict
from voicelens.nlp_models.entities import run_enr_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ------------------------------
# GCS MODEL PATHS
# ------------------------------
SENTIMENT_MODEL_URL = "https://storage.googleapis.com/voicelens/sentiment_model_pkl/2025-11-25_19-06-15_model_07_svm_linear_acc_0.9164.pkl"
TOPIC_MODEL_URL = (
    "https://storage.googleapis.com/voicelens/topics_model_pkl/topic_model_ismail.pkl"
)

# ------------------------------
# GLOBAL MODELS (loaded once)
# ------------------------------
sentiment_model = None
topic_model = None
nlp = None  # spaCy transformer model


def download_and_load_pickle(url: str):
    """Download a pickle file from GCS and load it into memory."""
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(response.content)
        tmp.flush()
        model = joblib.load(tmp.name)

    return model


@app.on_event("startup")
def load_models():
    # global sentiment_model, topic_model, nlp
    global sentiment_model, nlp

    print("Downloading and loading Sentiment model...")
    sentiment_model = download_and_load_pickle(SENTIMENT_MODEL_URL)

    # print("Downloading and loading Topic model...")
    # topic_model = download_and_load_pickle(TOPIC_MODEL_URL)

    print("Loading spaCy Transformer model: en_core_web_trf...")
    nlp = spacy.load("en_core_web_trf")  # Make sure this is in requirements.txt

    print("All models loaded successfully.")


@app.get("/")
def root():
    return {"message": "The API is running and models are loaded!"}


# ------------------------------
# PREDICTION ENDPOINT
# ------------------------------


class PredictRequest(BaseModel):
    reviews: list[str]


@app.post("/predict")
def predict_review(request: PredictRequest):
    # global sentiment_model, topic_model, nlp
    global sentiment_model, nlp

    results = []

    for review in request.reviews:

        # ------- Sentiment Prediction -------
        sentiment_pred = sentiment_model.predict([review])[0]

        # # ------- Topic Prediction -------
        # # Example: If topic_model is something like TfidfVectorizer + SVM
        # topic_pred = topic_model.predict([review])[0]

        # # ------- Entity Extraction Pipeline -------
        # df = pd.DataFrame([review], columns=["_clean_review"])

        # You can add your cleaning if needed:
        # For a single review
        # comment or "_clean_text" if you prefer
        df = pd.DataFrame(
            [review], columns=["comment"]
        )
        df = apply_cleaning(df, text_col="comment")
        cleaned_text = df["_clean_text"].iloc[0]

        # df["_clean_text"] = df["_clean_text"].apply(apply_cleaning)

        # Run pipeline
        enr_df = run_enr_model(df, text_col="_clean_text")
        entities = enr_df["entities"].iloc[0]

        results.append(
            {
                "text": review,
                "sentiment": sentiment_pred,
                "entities": entities,
            }
        )

    return results


# FOR ONE REVIEW PREDICTION
# class ReviewInput(BaseModel):
#     text: str

# @app.post("/predict")
# def predict_review(input: ReviewInput):
#     # global sentiment_model, topic_model, nlp
#     global sentiment_model, nlp

#     text = input.text

#     # ------- Sentiment Prediction -------
#     sentiment_pred = sentiment_model.predict([text])[0]

#     # # ------- Topic Prediction -------
#     # # Example: If topic_model is something like TfidfVectorizer + SVM
#     # topic_pred = topic_model.predict([text])[0]

#     # # ------- Entity Extraction Pipeline -------
#     # df = pd.DataFrame([text], columns=["_clean_text"])

#     # You can add your cleaning if needed:
#     # For a single review
#     df = pd.DataFrame([text], columns=["comment"])  # or "_clean_text" if you prefer
#     df = apply_cleaning(df, text_col="comment")
#     cleaned_text = df["_clean_text"].iloc[0]

#     # df["_clean_text"] = df["_clean_text"].apply(apply_cleaning)

#     # Run pipeline
#     enr_df = run_enr_model(df, text_col="_clean_text")
#     entities = enr_df["entities"].iloc[0]

#     return {
#         "text": text,
#         "sentiment": sentiment_pred,
#         # "topic": topic_pred,
#         "entities": entities,
#     }


# # Endpoint for https://your-domain.com/
# @app.get("/")
# def root():
#     return {"message": "The API is running!"}


# Load Models at Startup (Global Variables)
# sentiment_model = joblib.load("models/sentiment.pkl")
# topic_model = joblib.load("models/topic.pkl")
# ner_model = spacy.load("models/ner_model")

# @app.post("/predict")
# def predict_review(input: ReviewInput):
#     # Run the pre-loaded models on the single input text

#     # sentiment = sentiment_model.predict([input.text])[0]
#     # topic = topic_model.transform([input.text])
#     # entities = ner_model(input.text)

#     # Mock return for now
#     return {
#         "text": input.text,
#         "sentiment": "negative",
#         "topic": "Shipping",
#         "entities": [
#             ("around 3-25$ amazon credit", "MONEY"),
#             ("iphone", "PRODUCT"),
#             ("google", "ORG"),
#             ("30 minutes later", "TIME"),
#             ("this day", "DATE"),
#         ],
#     }


# # Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
# @app.get("/predict")
# def get_predict(
#     input_one: float,
#     input_two: float,
# ):
#     # TODO: Do something with your input
#     # i.e. feed it to your model.predict, and return the output
#     # For a dummy version, just return the sum of the two inputs and the original inputs
#     prediction = float(input_one) + float(input_two)
#     return {
#         "prediction": prediction,
#         "inputs": {"input_one": input_one, "input_two": input_two},
#     }
