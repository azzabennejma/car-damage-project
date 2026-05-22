FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v "^torch\|^torchvision\|^numpy" requirements.txt > requirements_filtered.txt \
    && pip install --no-cache-dir --timeout=300 -r requirements_filtered.txt \
    && pip install --no-cache-dir "numpy<2"

COPY main.py .

RUN mkdir -p model outputs/images outputs/labels outputs/annotated uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]