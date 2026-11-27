# sn85up — routed Real-ESRGAN specialists for SN85 generative upscaling

Entry for the VidAIO (SN85) generative upscaling competition (`sn85-upscaling-006`): ×2 (960×540 → 1920×1080) and ×4
(480×270 → 1920×1080), landscape and portrait.

## How it works

`/bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR` runs one Python process per batch (`app/sn85up/main.py`):

1. **Route.** A linear classifier (`router_v2`) on CLIP RN50 embeddings of 6 frames decides *screen* vs *natural*
   (3D renders and AI video) for each clip (`app/sn85up/router.py`).
2. **Upscale.** One of four fine-tuned Real-ESRGAN (RRDBNet) specialists — (×2, ×4) × (screen, natural) — upscales the
   clip on the GPU: bf16, channels-last, batches of 4 frames; the ×4 natural specialist also reads the previous and
   next frame (`tframes = 3`).
3. **Gate guard.** The validator's low-resolution consistency gate is measured on the GPU while upscaling; a clip that
   would fail it is nudged back toward the input (`app/sn85up/gate.py`).
4. **Encode.** Frames are converted to BT.709 limited-range 4:2:0 on the GPU and piped to libx264, lossless (`-qp 0`,
   falling back to CRF 4 / 8 only if a file would exceed the size caps), tagged BT.709 tv.

For speed, models load once per batch and only when needed, the next clips decode on CPU threads while the GPU
works, and encoding runs in parallel threads. Any failure of a clip falls back to a Lanczos upscale of that clip, and
the process always exits 0.

## Layout

| Path | What |
| --- | --- |
| `Dockerfile` | digest-pinned PyTorch 2.8 / CUDA 12.8 + static ffmpeg 9.0; downloads the weights (sha256-checked) and self-tests at build time |
| `run.sh` | the competition entry point |
| `app/sn85up/` | the runner, router, RRDBNet and CLIP RN50 code |
| `app/weights/router_v2.pt` | router weights (the specialists and CLIP are downloaded by the Dockerfile) |
| `tools/selftest.py` | build-time check that every model loads and runs |

Licences: `LICENSE` (MIT) and `LICENSES.md` (every third-party component and the training data).
