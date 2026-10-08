# The contract review agent. Build from the repo root, with data/raw/CUAD_v1.json in place.
#   docker build -t contract-agent .
#   docker run -p 8080:8080 -e OPENAI_API_KEY=... -v agent-data:/data contract-agent
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf PYTHONPATH=/app/src AGENT_DATA_DIR=/data KG_CUAD_PATH=/app/data/raw/CUAD_v1.json

WORKDIR /app

# The CPU build of torch is a few hundred MB smaller than the default one.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_lg

# Download the two retrieval models into the image, so the container does not need the network to start.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src ./src
COPY data/raw/CUAD_v1.json ./data/raw/CUAD_v1.json

# Run as a normal user. /data holds the SQLite files, so mount a volume there to keep them.
RUN useradd --create-home --uid 1000 app && mkdir /data && chown -R app /data /opt/hf
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"
CMD ["uvicorn", "agent.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080"]
