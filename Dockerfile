# Use a slim Python base with pip
FROM python:3.10-slim

# 1) Create app directory
WORKDIR /app

# 2) Copy app code + SavedModel dirs
COPY app.py .
COPY toxicity/ ./toxicity/
COPY vectorizer/ ./vectorizer/

# 3) Install runtime dependencies
RUN pip install --no-cache-dir \
      fastapi \
      uvicorn[standard] \
      tensorflow \
      nltk

# 4) Pre-download NLTK data so it’s baked into the image
RUN python - <<EOF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
EOF

# 5) Expose port and launch
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]