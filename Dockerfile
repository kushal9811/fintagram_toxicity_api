FROM python:3.10-slim

WORKDIR /app

# Copy code and TFLite models
COPY app.py .
COPY toxicity/ toxicity/
COPY vectorizer/ vectorizer/
COPY requirements.txt .

# Install minimal deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]