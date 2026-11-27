# SN85 generative upscaling entry: four Real-ESRGAN (RRDBNet) specialists (x2 / x4  x  screen / natural) behind a
# CLIP RN50 content router, run as one pipelined process per batch (see app/sn85up/main.py).
#
# Builder rules (docs/MINING.md): plain FROM / COPY / RUN / ENV / WORKDIR only; no ADD; image refs carry a digest
# OR a tag; build within 30 min without a GPU; image <= 40 GB. Every download is checked by sha256 in its RUN step.

# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (PyTorch 2.8.0, CUDA 12.8, Python 3.11), pinned by digest
FROM pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

# ffmpeg + ffprobe 9.0, static build with libx264 (mwader/static-ffmpeg:9.0), pinned by digest
COPY --from=mwader/static-ffmpeg@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- weights (downloaded at build time, sha256-checked in the same RUN step) ---------------------------------------
# The four specialists: our fine-tunes of Real-ESRGAN (BSD-3-Clause), released under this repository's licence.
# WEIGHTS_URL: replace with the published location of the four files (e.g. a Hugging Face repo pinned to a commit:
# https://huggingface.co/<owner>/<repo>/resolve/<40-hex commit>).
RUN mkdir -p /app/weights && cd /app/weights \
 && W=https://huggingface.co/OWNER/REPO/resolve/0000000000000000000000000000000000000000 \
 && for f in x2_natural x2_screen x4_natural x4_screen; do \
      python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$W/$f.safetensors" "$f.safetensors" || exit 1; \
    done \
 && printf '%s\n' \
      "7ac9b2fb99cb3e303acd7a5fdaa960fec91f80606f43a3f7a06effa0fa60cd21  x2_natural.safetensors" \
      "a598be655a8c02f1acf2654421dda43c2cda2909f019a16bdf3dc18d2829c0d0  x2_screen.safetensors" \
      "97cd97c736137faacc5db706464bc4d4145e38c81703bf3dd2dc070bee51a257  x4_natural.safetensors" \
      "5ae044ad6e8a422547ea48d465453dac2bff23872bad7081a6cb6b96fea73a8e  x4_screen.safetensors" \
    | sha256sum -c -

# CLIP RN50 (OpenAI, MIT): the original release, the same file the validator's CLIP-IQA uses.
RUN cd /app/weights \
 && python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt" clip-RN50.pt \
 && echo "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762  clip-RN50.pt" | sha256sum -c -
# --------------------------------------------------------------------------------------------------------------------

COPY run.sh /app/run.sh
COPY app/sn85up /app/sn85up
COPY app/weights/router_v2.pt /app/weights/router_v2.pt
COPY tools/selftest.py /app/tools/selftest.py

# TorchScript CLIP -> plain state dict of the visual tower (fast weights_only load at run time), then fail the BUILD,
# not the evaluation, if anything the run needs is missing: ffmpeg/libx264, timeout, every model loads and runs.
RUN chmod 0755 /app/run.sh \
 && cd /app && python -c "from sn85up.clip_rn50 import convert_jit; convert_jit('weights/clip-RN50.pt', 'weights/clip_rn50_visual.pt')" \
 && rm -f /app/weights/clip-RN50.pt \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
 && ffprobe -version >/dev/null && command -v timeout >/dev/null \
 && PYTHONPATH=/app python /app/tools/selftest.py
