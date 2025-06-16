from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Import Interpreter and load_delegate from TF Python
from tensorflow.lite.python.interpreter import Interpreter, load_delegate

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Path helper
def make_interp(path: str):
    return Interpreter(
        model_path=path,
        experimental_delegates=[load_delegate("libtensorflowlite_flex.so")]
    )

# Load TFLite models with Flex delegate
tox_interp = make_interp("toxicity/toxicity.tflite")
tox_interp.allocate_tensors()
tox_in  = tox_interp.get_input_details()[0]["index"]
tox_out = tox_interp.get_output_details()[0]["index"]

vec_interp = make_interp("vectorizer/vectorizer.tflite")
vec_interp.allocate_tensors()
vec_in  = vec_interp.get_input_details()[0]["index"]
vec_out = vec_interp.get_output_details()[0]["index"]

def vectorize(text: str) -> np.ndarray:
    vec_interp.set_tensor(vec_in, np.array([text], dtype=object))
    vec_interp.invoke()
    return vec_interp.get_tensor(vec_out).astype(np.float32)

def score_comment(text: str, threshold: float = 0.5) -> dict:
    vect = vectorize(text)
    tox_interp.set_tensor(tox_in, vect)
    tox_interp.invoke()
    probs = tox_interp.get_tensor(tox_out)[0]

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
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/score")
def score_endpoint(req: In):
    return score_comment(req.text)

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)