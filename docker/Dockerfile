# The job image for a Scaleway GPU machine: the pinned model libraries
# (`miroir.jobs.remote_spec.MODEL_LIBRARIES`) and the miroir package, serving the jobs on one
# port (`miroir scw serve`). Built on the machine itself by `miroir.jobs.scw_run`, which writes
# requirements.txt from those pins and picks TORCH_INDEX to match the machine's NVIDIA driver.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.29

ARG TORCH_INDEX
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1
COPY requirements.txt match.txt lightglue.txt /build/
# torch first, from the index built for the machine's CUDA; the rest from PyPI keeps it.
RUN grep -E '^(torch|torchvision)==' /build/requirements.txt > /build/torch.txt \
    && uv pip install --index-url "$TORCH_INDEX" -r /build/torch.txt \
    && uv pip install -r /build/requirements.txt
# Keypoint matching (MATCH=1) only: OpenCV pulls numpy below the embedding image's pin, so
# the embedding image leaves it out.
ARG MATCH=0
RUN if [ "$MATCH" = "1" ]; then \
        uv pip install -r /build/match.txt && uv pip install --no-deps -r /build/lightglue.txt; \
    fi

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
RUN uv pip install --no-deps /app

# Model caches live in /weights, mirrored from the weights bucket.
ENV HF_HOME=/weights/hf TORCH_HOME=/weights/torch PYTHONUNBUFFERED=1
ENTRYPOINT ["miroir", "scw", "serve"]
