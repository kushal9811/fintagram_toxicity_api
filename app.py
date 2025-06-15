# app.py

# 1) Imports
import re
import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 2) Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# 3) Load the vectorizer and model (SavedModel folders in same dir)
vectorizer = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.TFSMLayer("vectorizer", call_endpoint="serving_default")
])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1800,), dtype=tf.float32),
    tf.keras.layers.TFSMLayer("toxicity", call_endpoint="serving_default")
])

# 4) Define your scoring function
def score_comment(text: str, threshold: float = 0.5) -> dict:
    # Vectorize and cast
    vect_dict = vectorizer(tf.constant([text]))
    vect = tf.cast(vect_dict["text_vectorization"], tf.float32)

    # Direct call to model returns dict
    output = model(vect, training=False)
    # Extract the first tensor in the returned dict
    probs_tensor = list(output.values())[0]
    probs = probs_tensor.numpy()[0]

    # Build response dict
    result = {}
    for label, p in zip(LABELS, probs):
        result[label] = float(p)
        result[f"{label}_flag"] = int(p > threshold)
    return result

# 5) FastAPI wiring
class In(BaseModel):
    text: str

app = FastAPI(title="Toxicity Scoring API")

@app.post("/score")
def score_endpoint(req: In):
    return score_comment(req.text)