# The voice studio and the `voiceclone` command in one image (linux/amd64, linux/arm64).
# CPU-only torch: Docker on a Mac cannot reach the Apple GPU, and the CUDA
# build would triple the size. Synthesis works on CPU but is slow (minutes per
# sentence); training needs a GPU or Apple silicon outside Docker.
#
#   docker run --rm -p 127.0.0.1:8000:8000 -v voiceclone:/data -e HF_TOKEN \
#     ghcr.io/svayatta/indic-voice-clone
#
# Everything the container writes (voices, the Hugging Face model cache,
# speaker and alignment models) goes under /data, so one volume keeps it.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
# ctc-forced-aligner ships as source with a C++ extension
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml constraints.txt README.md ./
COPY src ./src
# torch and torchaudio first, from the CPU index; the constraints then pin
# everything else to the tested versions (torch==2.14.1 matches 2.14.1+cpu).
RUN uv venv /opt/venv --python /usr/local/bin/python3.12 \
 && uv pip install --python /opt/venv/bin/python --no-cache torch==2.14.1 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --python /opt/venv/bin/python --no-cache ".[all]" -c constraints.txt

FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --home-dir /data --uid 1000 voiceclone
USER voiceclone
ENV HOME=/data HF_HOME=/data/hf PYTHONUNBUFFERED=1
WORKDIR /data
VOLUME /data
EXPOSE 8000

ENTRYPOINT ["voiceclone"]
# 0.0.0.0 inside the container; publish it as 127.0.0.1:8000 on the host to keep it local
CMD ["serve", "--host", "0.0.0.0", "--root", "/data/studio"]
