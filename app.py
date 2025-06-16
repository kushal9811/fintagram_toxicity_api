# app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tflite_runtime.interpreter as tflite

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Load TFLite toxicity model
tox_interp = tflite.Interpreter(model_path="toxicity/toxicity.tflite")
tox_interp.allocate_tensors()
tox_in  = tox_interp.get_input_details()[0]["index"]
tox_out = tox_interp.get_output_details()[0]["index"]

# Load TFLite vectorizer model
vec_interp = tflite.Interpreter(model_path="vectorizer/vectorizer.tflite")
vec_interp.allocate_tensors()
vec_in  = vec_interp.get_input_details()[0]["index"]
vec_out = vec_interp.get_output_details()[0]["index"]

def vectorize(text: str) -> np.ndarray:
    # TFLite string inputs use dtype=object
    vec_interp.set_tensor(vec_in, np.array([text], dtype=object))
    vec_interp.invoke()
    return vec_interp.get_tensor(vec_out).astype(np.float32)

def score_comment(text: str, threshold: float = 0.5) -> dict:
    vect = vectorize(text)                # shape (1, seq_len)
    tox_interp.set_tensor(tox_in, vect)
    tox_interp.invoke()
    probs = tox_interp.get_tensor(tox_out)[0]  # (6,)

    result = {}
    for label, p in zip(LABELS, probs):
        result[label]        = float(p)
        result[f"{label}_flag"] = int(p > threshold)
    return result

class In(BaseModel):
    text: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # lock down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/score")
def score_endpoint(req: In):
    return score_comment(req.text)