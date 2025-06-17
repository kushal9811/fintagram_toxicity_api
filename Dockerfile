FROM python:3.10-slim
WORKDIR /app

COPY app.py .
COPY toxicity/ ./toxicity/
COPY vectorizer/ ./vectorizer/

RUN pip install --no-cache-dir fastapi uvicorn[standard] tensorflow
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]