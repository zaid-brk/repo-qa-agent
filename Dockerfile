FROM python:3.11-slim

# repos are cloned via git during indexing (see ingest.py)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render (and HF Spaces) inject the port to bind via $PORT; default 7860 for local docker run
ENV PORT=7860
EXPOSE 7860
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
