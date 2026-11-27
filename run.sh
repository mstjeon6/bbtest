#!/bin/sh
# Competition contract (upscaling track, generative scoring): /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
#   - every regular, non-hidden file in INPUT_DIR is one item (named by its sha256, no extension); the hidden file
#     INPUT_DIR/.vidaio-next-upscale-task-<name> holds {"upscale_factor", "target_width", "target_height"};
#   - one MP4 per item in OUTPUT_DIR under the same name: one H.264 stream, target size, input frame count and rate;
#   - no network, no environment, fresh sandbox per batch; a non-zero exit or a timeout zeroes the whole batch.
# The whole batch runs in ONE Python process (models load once). It falls back to Lanczos per clip on any failure
# and always exits 0; the outer timeout is a last resort below the manifest's 3600 s batch timeout.
# No `set -e` on purpose.
set -u
umask 022
[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
PATH=/usr/local/bin:/opt/conda/bin:/usr/bin:/bin
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=8
export PATH PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED OMP_NUM_THREADS
python=${PYTHON:-/opt/conda/bin/python}
mkdir -p "$2"
if command -v timeout >/dev/null 2>&1; then
    timeout -k 15 3450 "$python" -m sn85up.main "$1" "$2"
else
    "$python" -m sn85up.main "$1" "$2"
fi
rc=$?
[ "$rc" -eq 0 ] || echo "sn85up exited $rc; outputs written so far are kept" >&2
# never leave temporary files behind
find "$2" -maxdepth 1 -name '.*' -type f -delete 2>/dev/null
exit 0
