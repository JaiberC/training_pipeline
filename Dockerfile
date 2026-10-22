# syntax=docker/dockerfile:1.7
# La imagen se construye UNA vez y se hace pull tanto en la Tesla (GPU) como en
# máquinas CPU-only. Por eso NO se hornea CUDA aquí: el dispositivo se elige en
# runtime con torch.cuda.is_available().
ARG PYTHON_VERSION=3.12
ARG VERSION=dev

# ── base ───────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
# libgl1/libglib2.0-0: opencv. tini: PID 1 para reaping y propagacion de
# SIGTERM, que es lo que hace funcionar `cancel`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv $VIRTUAL_ENV

# ── deps: capa cacheada, se invalida solo al tocar pyproject.toml ───────
FROM base AS deps-api
WORKDIR /app
COPY pyproject.toml README.md ./
# Stub vacio: setuptools exige que exista el directorio del paquete. NO se copia
# el codigo real aqui a proposito, para que un cambio en src/ no invalide esta
# capa. En runtime PYTHONPATH=/app/src tiene precedencia sobre site-packages,
# asi que el codigo de /app/src gana al stub instalado.
RUN mkdir -p src/trainpipe && touch src/trainpipe/__init__.py
RUN pip install --no-cache-dir ".[api]"

FROM base AS deps-worker
WORKDIR /app
COPY pyproject.toml README.md ./
RUN mkdir -p src/trainpipe && touch src/trainpipe/__init__.py
RUN pip install --no-cache-dir ".[worker]"

# ── runtime-api: sin torch ─────────────────────────────────────────────
FROM base AS runtime-api
ARG VERSION
LABEL org.opencontainers.image.version=$VERSION
WORKDIR /app
COPY --from=deps-api /opt/venv /opt/venv
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
COPY docker/entrypoint-api.sh /usr/local/bin/entrypoint-api.sh
RUN chmod +x /usr/local/bin/entrypoint-api.sh
ENV PYTHONPATH=/app/src
EXPOSE 8000
# El entrypoint aplica `alembic upgrade head` antes de arrancar uvicorn, de modo
# que un despliegue limpio crea el esquema solo. El worker depende de que esta
# API este sana, y su salud ahora significa "esquema listo".
ENTRYPOINT ["/usr/local/bin/entrypoint-api.sh"]
CMD ["uvicorn", "trainpipe.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── runtime-worker: con torch ──────────────────────────────────────────
FROM base AS runtime-worker
ARG VERSION
LABEL org.opencontainers.image.version=$VERSION
WORKDIR /app
# gcc/g++: Triton compila los kernels de GPU en tiempo de ejecucion y para eso
# necesita un compilador de C/C++. Sin el, un entrenamiento con
# `yolo26s-seg.pt` muere al primer batch con
# `RuntimeError: Failed to find C compiler`, y el error que llega al usuario es
# un `EOFError` del hilo de pin_memory, que no dice nada de la causa real.
#
# Solo va en el worker: la imagen de la API no lleva torch y no lo necesita.
# Va ANTES de los COPY para que un cambio en src/ no vuelva a ejecutar
# apt-get (y con ello vuelva a necesitar red).
#
# Verificado: con deteccion (yolo26n, yolo26s) el entrenamiento completa sin
# compilador. Solo la segmentacion dispara Triton.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY --from=deps-worker /opt/venv /opt/venv
COPY src ./src
ENV PYTHONPATH=/app/src
ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "trainpipe.worker"]