FROM python:3.10-slim

WORKDIR /app

# 1) Pre‑install numpy<2 and a compatible tflite‑runtime
RUN pip install --no-cache-dir \
      "numpy<2" \
      "tflite-runtime==2.13.0"

# 2) Install any other Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your code and models
COPY app.py .
COPY toxicity/ toxicity/
COPY vectorizer/ vectorizer/

# 4) Expose the port (Render will set $PORT)
EXPOSE 8000

# 5) Run via Python so app.py's __main__ block uses the $PORT env var
CMD ["python", "app.py"]