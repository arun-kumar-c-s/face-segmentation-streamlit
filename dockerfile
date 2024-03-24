FROM python:3.9

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt

# Download the model files and cache them in the Docker image
RUN mkdir -p /root/.cache/huggingface/hub/models--jonathandinu--face-parsing
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jonathandinu/face-parsing', cache_dir='/root/.cache/huggingface/hub/models--jonathandinu--face-parsing')"
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.9

# WORKDIR /app

# COPY requirements-backend.txt .
# RUN pip install --no-cache-dir -r requirements-backend.txt

# # Download the model files and cache them in the Docker image
# RUN mkdir -p /root/.cache/huggingface/hub/models--jonathandinu--face-parsing
# RUN python -c "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation; SegformerImageProcessor.from_pretrained('jonathandinu/face-parsing'); SegformerForSemanticSegmentation.from_pretrained('jonathandinu/face-parsing')"

# COPY main.py .

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]