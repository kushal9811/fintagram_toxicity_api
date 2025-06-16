FROM python:3.10-slim

WORKDIR /app

# 1) Install numpy <2 and tflite‐runtime
RUN pip install --no-cache-dir \
      "numpy<2" \
      "tflite-runtime==2.13.0"

# 2) Install your other Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy only the .tflite files and your server code
COPY toxicity/toxicity.tflite    toxicity/
COPY vectorizer/vectorizer.tflite vectorizer/
COPY app.py .

# 4) Expose port and launch Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]