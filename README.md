# Finstagram Toxicity API

This repository contains a complete solution for building, packaging, and deploying a comment toxicity scoring service. It uses:

* A **Bidirectional LSTM** model trained on the Jigsaw Toxic Comment Classification dataset (TensorFlow SavedModel)
* **Docker** to containerize the FastAPI server and TensorFlow model
* **GitHub Actions** CI/CD pipeline for automated build, smoke test, and Docker Hub push
* **Render** for hosting the API as a web service

---

## 📂 Repository Structure

```
├── .github/workflows/ci.yml      # GitHub Actions CI/CD workflow
├── toxic-comment-classification/
│   └── dataset files
├── toxicity/                     # LSTM SavedModel directory
│   ├── saved_model.pb
│   ├── variables/
│   └── assets/
├── vectorizer/                   # TextVectorization SavedModel directory
│   ├── saved_model.pb
│   ├── variables/
│   └── assets/
├── app.py                        # FastAPI server loading SavedModels via TFSMLayer
├── Dockerfile                    # Container build instructions
├── requirements.txt              # Python dependencies (TensorFlow, FastAPI, etc.)
├── toxic-comments-classification-wth-lstm-based-model.ipynb # Notebook and training code (LSTM model)
└── test_toxicity.py              # Local inference test script
```

---

## 🧠 LSTM Model (Notebook)

The `toxic-comments-classification-wth-lstm-based-model.ipynb` notebook contains:

1. **Data loading & preprocessing** using `TextVectorization` and tokenization
2. **Bidirectional LSTM** architecture with embedding, dropout, and dense layers
3. **Training** on the Jigsaw dataset and evaluation metrics (precision, recall, accuracy)
4. **Export** of the trained model as a TensorFlow SavedModel (`toxicity/` folder)

---

## 🚀 FastAPI Server (`app.py`)

* Loads the two SavedModels (vectorizer & toxicity) via `tf.keras.layers.TFSMLayer`
* Defines a `POST /score` endpoint that accepts `{ text: string }`
* Returns probabilities and binary flags for each toxicity label

**Local run:**

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Container

**Dockerfile**:

1. Starts from `python:3.10-slim`
2. Installs `numpy<2` and `tensorflow`; pins versions for consistency
3. Installs other dependencies from `requirements.txt`
4. Copies models and source code
5. Runs `python app.py`, which reads `$PORT` for Render

**Build & run locally:**

```bash
docker build -t kushal88/toxicity-api:latest .
docker run -p 8000:8000 kushal88/toxicity-api:latest
```

---

## 🔄 CI/CD with GitHub Actions

File: `.github/workflows/ci.yml`:

1. **Build** the Docker image
2. **Smoke test** by running the container, dumping logs, and curling `/score`
3. **Login** to Docker Hub (using secrets)
4. **Push** the image to Docker Hub

Every push to `main` triggers this workflow.

---

## 🌐 Deployment on Render

1. **Create** a new Web Service on Render, connect your GitHub repo
2. **Environment**: Docker
3. **Start command**: `python app.py`

   * Render sets the `$PORT` environment variable automatically
4. **Plan**: Free / Starter (512 MB RAM) – upgrade if OOM
5. **Logs**: Check the Render dashboard for runtime errors or memory usage

---

## 🎯 Usage

```bash
curl --location 'https://toxicity-api-kushal.onrender.com/score' \
--header 'Content-Type: application/json' \
--data '{"text":"you suck man"}'
```

Response:

```json
{
  "toxic": 0.99,
  "toxic_flag": 1,
  "severe_toxic": 0.15,
  "severe_toxic_flag": 0,
  ...
}
```

---

## 📄 License

MIT © Kushal9811
