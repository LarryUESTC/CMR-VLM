# CMR-VLM clean-install image with pinned dependency versions.
#
# The image runs all patient-free smoke tests:
#   1. python demo.py                              (lightweight CPU forward)
#   2. reproducibility: make test                  (6/6 verifier unit tests)
#   3. reproducibility: make verify                (21/21 checkpoint-header checks)
#   4. reproducibility: make full-smoke            (strict full checkpoint load + synthetic encoder forward)
#
# Steps 3 and 4 need the released checkpoint mounted at /data/output (or set
# CMR_VLM_CHECKPOINT to another in-container path).
#
# Build (CPU-only; no NVIDIA driver required):
#   docker build -t cmr-vlm:smoke .
#
# Run the install + demo + verifier smoke tests:
#   docker run --rm cmr-vlm:smoke
#
# Run everything including the full checkpoint load (bind-mount the checkpoint):
#   docker run --rm -e CMR_VLM_SKIP_FULL_LOAD=0 \
#       -v /path/to/output:/data/output:ro cmr-vlm:smoke

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

WORKDIR /app

# 1) PyTorch CPU wheels from the official CPU index (pinned exact +cpu builds;
#    --extra-index-url keeps PyPI as the primary index for all other packages).
RUN pip install \
    torch==2.9.1+cpu torchvision==0.24.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2) Minimal pinned runtime dependencies for the smoke tests.
COPY requirements-smoke.txt ./
RUN pip install -r requirements-smoke.txt

# 3) Repository code (model sources, verifier package, demo).
COPY . .

ENTRYPOINT ["bash", "docker/smoke.sh"]
