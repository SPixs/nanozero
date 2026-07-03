# Image du jobserver NanoZero — FastAPI + uvicorn + PostgreSQL (psycopg3).
# Build context = nanozero-jobserver/ (cf. .dockerignore : la DB et les datasets
# sont EXCLUS du contexte). Config 100 % via variables d'environnement
# (cf. src/nanozero_jobserver/config.py). La base vit dans le service `postgres` ;
# seuls les shards NPZ sont écrits sur le volume /data.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dépendances + build du package via PEP 517 (backend poetry-core, déclaré dans
# pyproject). numpy/pydantic-core/uvloop/httptools ET psycopg[binary] (libpq
# bundlée) ont des wheels manylinux → pas de toolchain C ni libpq-dev requis.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Volume pour les shards NPZ uniquement (npz_output_dir = ./datasets → /data/datasets).
# La base PostgreSQL est dans le service `postgres` (volume postgres-data).
VOLUME ["/data"]
WORKDIR /data

EXPOSE 8090

# cli() lit toute la config depuis l'env (host/port défaut 0.0.0.0:8090, pas d'argv).
# Docker envoie SIGTERM à l'arrêt → shutdown gracieux (force_flush du tampon flotte).
CMD ["nanozero-jobserver"]
