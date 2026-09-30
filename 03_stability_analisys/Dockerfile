# -----------------------------------------------------------------------------
# Build GROMACS 2025.4 (CPU-only; sufficient for trjconv preprocessing)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS gromacs-builder

ARG GROMACS_VERSION=2025.4
ARG GROMACS_URL=https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz

ARG CMAKE_VERSION=3.31.6

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libfftw3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "cmake==${CMAKE_VERSION}" \
    && cmake --version

RUN curl -fsSL "${GROMACS_URL}" -o /tmp/gromacs.tar.gz \
    && tar -xzf /tmp/gromacs.tar.gz -C /tmp \
    && cmake \
        -S "/tmp/gromacs-${GROMACS_VERSION}" \
        -B /tmp/gromacs-build \
        -DCMAKE_INSTALL_PREFIX=/opt/gromacs \
        -DCMAKE_BUILD_TYPE=Release \
        -DGMX_GPU=OFF \
        -DGMX_MPI=OFF \
        -DGMX_OPENMP=ON \
        -DGMX_BUILD_OWN_FFTW=OFF \
        -DREGRESSIONTEST_DOWNLOAD=OFF \
        -DBUILD_TESTING=OFF \
    && cmake --build /tmp/gromacs-build --parallel "$(nproc)" \
    && cmake --install /tmp/gromacs-build \
    && rm -rf /tmp/gromacs.tar.gz "/tmp/gromacs-${GROMACS_VERSION}" /tmp/gromacs-build

# -----------------------------------------------------------------------------
# Runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libfftw3-single3 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gromacs-builder /opt/gromacs /opt/gromacs

ENV PATH=/opt/gromacs/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    HOME=/tmp \
    GMX_MAXBACKUP=-1

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
