# TODO: Import your package, replace this by explicit imports of what you need
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from voicelens.main import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {"message": "The API is running!"}


# Load Models at Startup (Global Variables)
# sentiment_model = joblib.load("models/sentiment.pkl")
# topic_model = joblib.load("models/topic.pkl")
# ner_model = spacy.load("models/ner_model")

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict_review(input: ReviewInput):
    # Run the pre-loaded models on the single input text

    # sentiment = sentiment_model.predict([input.text])[0]
    # topic = topic_model.transform([input.text])
    # entities = ner_model(input.text)

    # Mock return for now
    return {
        "text": input.text,
        "sentiment": "Negative",
        "topic": "Shipping",
        "entities": [{"text": "FedEx", "label": "ORG"}]
    }

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
