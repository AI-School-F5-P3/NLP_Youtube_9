FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/usr/local/share/nltk_data\
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data with error handling
RUN mkdir -p /usr/local/share/nltk_data && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab && \
    python -m nltk.downloader -d /usr/local/share/nltk_data stopwords && \
    python -m nltk.downloader -d /usr/local/share/nltk_data wordnet && \
    python -m nltk.downloader -d /usr/local/share/nltk_data averaged_perceptron_tagger && \
    python -c "import nltk; print('NLTK Data downloaded successfully')"

# Copy the Firebase credentials
COPY secrets/serviceAccountKey.json /app/serviceAccountKey.json

# Copy application code
COPY . .

# Expose the dynamic port specified by Cloud Run
EXPOSE $PORT

# Create run script to use the Cloud Run port environment variable
RUN echo '#!/bin/bash\nset -e\nstreamlit run GUI.py --server.port $PORT --server.address 0.0.0.0' > run.sh && \
    chmod +x run.sh

# Add healthcheck
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

# Run the application
CMD ["./run.sh"]