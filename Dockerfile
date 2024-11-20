FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NLTK_DATA=/usr/local/share/nltk_data\
    PORT=8080

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/local/share/nltk_data && \
    python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab && \
    python -m nltk.downloader -d /usr/local/share/nltk_data stopwords && \
    python -m nltk.downloader -d /usr/local/share/nltk_data wordnet && \
    python -m nltk.downloader -d /usr/local/share/nltk_data averaged_perceptron_tagger && \
    python -c "import nltk; print('NLTK Data downloaded successfully')"

COPY secrets/serviceAccountKey.json /app/serviceAccountKey.json

COPY . .

EXPOSE $PORT

RUN echo '#!/bin/bash\nset -e\nstreamlit run GUI.py --server.port $PORT --server.address 0.0.0.0' > run.sh && \
    chmod +x run.sh

HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health || exit 1

CMD ["./run.sh"]