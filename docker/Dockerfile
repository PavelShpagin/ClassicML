FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && \
	apt-get install -y --no-install-recommends build-essential && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas==2.2.2 numpy==1.26.4

COPY src ./src
COPY scripts ./scripts
COPY README.md ./README.md
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN mkdir -p /app/submissions && chmod +x /app/docker/entrypoint.sh

# Expect host data mounted at /app/data/raw
VOLUME ["/app/data/raw", "/app/submissions"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]



