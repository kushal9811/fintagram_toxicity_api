FROM python:3.10-slim
WORKDIR /app

# 1) Copy code
COPY app.py .
COPY toxicity/ ./toxicity/
COPY vectorizer/ ./vectorizer/

# 2) Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] tensorflow nltk

# 3) Bake in NLTK corpora (at image build time)
RUN python - <<EOF
import nltk
nltk.download('punkt',     download_dir='/usr/share/nltk_data')
nltk.download('stopwords', download_dir='/usr/share/nltk_data')
nltk.download('wordnet',   download_dir='/usr/share/nltk_data')
EOF

# 4) Point NLTK to the baked‑in data
ENV NLTK_DATA=/usr/share/nltk_data

# 5) Expose port & start command
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]