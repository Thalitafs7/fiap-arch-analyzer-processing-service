## Stage 1 — builder: instala deps com toolchain de compilação
FROM python:3.12-slim AS builder

WORKDIR /app

# Toolchain só no builder (psycopg2, build wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala torch CPU-only (evita CUDA ~2GB)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type f -name "*.pyc" -delete

## Stage 2 — runtime: imagem mínima
FROM python:3.12-slim

WORKDIR /app

# Runtime libs (libpq para psycopg, sem gcc/build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copia deps já instaladas do builder
COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
