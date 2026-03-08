FROM python:3.11-slim

WORKDIR /app

# Install system deps for numpy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Install Python deps (lean server-only requirements, no pygame/torch)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application
COPY . .

# Cloud Run sets PORT env var; default to 8080
ENV PORT=8080

CMD uvicorn server:app --host 0.0.0.0 --port $PORT
