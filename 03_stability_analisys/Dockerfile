FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    HOME=/tmp

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.13.1 \
        pandas==2.2.3 \
        matplotlib==3.9.2 \
        MDAnalysis==2.10.0 \
        pymbar==4.0.3 \
        statsmodels==0.14.4

WORKDIR /app
COPY rmsd_plateau.py /app/rmsd_plateau.py

ENTRYPOINT ["python", "/app/rmsd_plateau.py"]
