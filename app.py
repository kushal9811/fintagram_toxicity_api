import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

vectorizer = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.TFSMLayer("vectorizer", call_endpoint="serving_default")
])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1800,), dtype=tf.float32),
    tf.keras.layers.TFSMLayer("toxicity", call_endpoint="serving_default")
])

def score_comment(text: str, threshold: float = 0.5) -> dict:
    vect_dict = vectorizer(tf.constant([text]))
    vect = tf.cast(vect_dict["text_vectorization"], tf.float32)
    output = model(vect, training=False)
    probs = list(output.values())[0].numpy()[0]

    return {
        **{label: float(p) for label, p in zip(LABELS, probs)},
        **{f"{label}_flag": int(p > threshold) for label, p in zip(LABELS, probs)}
    }

class In(BaseModel):
    text: str

app = FastAPI(title="Toxicity Scoring API")

@app.post("/score")
def score_endpoint(req: In):
    return score_comment(req.text)