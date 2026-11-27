"""One run.sh call = one batch: route every clip, upscale it with its specialist, encode, verify.

  python -m sn85up.main INPUT_DIR OUTPUT_DIR

Pipeline (the GPU never waits on I/O): a thread pool decodes the next clips into memory (ffmpeg -> rgb24) while the
GPU upscales the current one; each clip's frames go through bf16 channels-last RRDBNet, are converted to BT.709
limited 4:2:0 on the GPU and are piped to an ffmpeg libx264 encoder running in its own thread. Models are loaded once
per batch, lazily (only the specialists this batch needs).

Specialists: (factor, route) -> weights file; route = "screen" or "natural" (3D renders and AI video), from the
router. Any failure of a clip (decode, model, encode, verification) falls back to a Lanczos upscale of that clip, so
one bad clip never costs the batch; the process always exits 0 once it has started.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

T0 = time.time()
APP = Path(__file__).resolve().parents[1]
WEIGHTS = APP / "weights"
SPECIALISTS = {
    (2, "natural"): "x2_natural.safetensors",
    (2, "screen"): "x2_screen.safetensors",
    (4, "natural"): "x4_natural.safetensors",
    (4, "screen"): "x4_screen.safetensors",
}
SIDECAR = ".vidaio-next-upscale-task-"
BATCH_BUDGET_S = 3300          # batch_timeout_seconds is 3600: leave a margin for the Lanczos fallbacks
PRESET = "veryfast"
# Encoding: lossless (x264 -qp 0) scores best (p02 0.526 vs 0.450 at CRF 10; p07 0.519 vs 0.470) at no time cost.
# Per clip the best rung whose conservative size estimate fits the caps: qp 0 (measured <= 0.72 MB per 1080p frame),
# then CRF 4 (<= 0.40 MB), then CRF 8.
LADDER = ((0, 1.2e6), (4, 0.6e6), (8, 0.35e6))   # (crf, estimated bytes per output frame); crf 0 = qp 0
MAX_FILE = 500_000_000         # 512 MiB per file
MAX_TOTAL = 1_900_000_000      # 2 GiB per batch at any moment
GPU_BATCH = 4                  # same throughput as 8 on an A100, half the activation memory
YUV2RGB = ("scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709:"
           "flags=accurate_rnd+full_chroma_int+bicubic,format=rgb24")
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
_END = object()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.2f}s] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ I/O helpers
def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", path],
                         check=True, capture_output=True, text=True, timeout=300).stdout
    s = json.loads(out)["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]), "rate": s["r_frame_rate"],
            "fps": str(Fraction(s["r_frame_rate"])), "packets": int(s.get("nb_read_packets") or 0)}


def read_task(path: Path, info: dict) -> tuple[int, int, int]:
    """(factor, target width, target height) from the hidden task file; inferred from the input size if absent."""
    try:
        doc = json.loads((path.parent / (SIDECAR + path.name)).read_text())
        return int(doc["upscale_factor"]), int(doc["target_width"]), int(doc["target_height"])
    except Exception as e:  # noqa: BLE001
        f = 4 if max(info["width"], info["height"]) <= 480 else 2
        log(f"{path.name[:12]}: no readable task file ({e!r}); assuming x{f}")
        return f, info["width"] * f, info["height"] * f


def decode(path: str, w: int, h: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "8", "-i", path, "-map", "0:v:0", "-vf", YUV2RGB,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True, timeout=600).stdout
    n = len(raw) // (w * h * 3)
    if n == 0:
        raise RuntimeError("decoded no frames")
    return np.frombuffer(bytearray(raw), np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)   # writable (pinned copy)


def decode_luma(path: str, w: int, h: int) -> np.ndarray:
    """The input's Y planes exactly as the validator reads them (yuv420p, no conversion): (T, h, w) uint8."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True, timeout=600).stdout
    fs = w * h * 3 // 2
    n = len(raw) // fs
    return np.frombuffer(raw[: n * fs], np.uint8).reshape(n, fs)[:, : w * h].reshape(n, h, w).copy()


def lanczos(src: str, dst: str, W: int, H: int, crf: int = 12) -> None:
    """The safe fallback: plain Lanczos upscale (scores low but passes the gates)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0", "-fps_mode", "passthrough",
           "-vf", f"scale={W}:{H}:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), *TAGS, "-an", "-f", "mp4", dst]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)


class Encoder(threading.Thread):
    """Raw 4:2:0 planes from a queue -> ffmpeg libx264 -> dst (an MP4 with one BT.709 tv-range H.264 stream)."""

    def __init__(self, dst: str, W: int, H: int, rate: str, crf: int):
        super().__init__(daemon=True)
        self.dst, self.q, self.err = dst, queue.Queue(), None   # unbounded: a clip's planes are already in RAM
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}",
               "-framerate", rate, "-i", "-", "-c:v", "libx264", "-preset", PRESET, *(["-qp", "0"] if crf == 0 else ["-crf", str(crf)]),
               "-pix_fmt", "yuv420p", *TAGS, "-an", "-f", "mp4", dst]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.start()

    def put(self, item) -> None:
        """Blocking put that gives up (raises) if the encoder process has died."""
        while True:
            if self.err:
                raise RuntimeError(self.err)
            try:
                self.q.put(item, timeout=1.0)
                return
            except queue.Full:
                if not self.is_alive():
                    raise RuntimeError(self.err or "encoder thread exited")

    def abort(self) -> None:
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.q.put_nowait(_END)
        except queue.Full:
            pass
        self.join(timeout=10)

    def run(self):
        try:
            while (item := self.q.get()) is not _END:
                for y, cb, cr in zip(*item):
                    self.p.stdin.write(y.tobytes()); self.p.stdin.write(cb.tobytes()); self.p.stdin.write(cr.tobytes())
            self.p.stdin.close()
            if self.p.wait() != 0:
                self.err = f"x264 exit {self.p.returncode}: {self.p.stderr.read()[-500:]!r}"
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)
            try:
                self.p.kill()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------------------------------------------------ GPU side
class Upscaler:
    def __init__(self):
        import torch
        self.torch = torch
        torch.backends.cudnn.benchmark = os.environ.get("SN85UP_CUDNN_BENCHMARK", "1") == "1"   # test knob only
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.models, self.router = {}, None

    def get_router(self):
        if self.router is None:
            from .router import Router
            self.router = Router(WEIGHTS, "cuda")
            log("router loaded")
        return self.router

    def get_model(self, key):
        if key not in self.models:
            from .rrdbnet import load_model
            self.models[key] = load_model(WEIGHTS / SPECIALISTS[key], "cuda").to(memory_format=self.torch.channels_last)
            log(f"specialist {key} loaded (tframes {self.models[key].tframes})")
        return self.models[key]

    def run(self, model, frames, y_in, enc: Encoder) -> tuple[int, float, float, float]:
        """frames: (T, 3, h, w) uint8 on the GPU, y_in: (T, h, w) uint8 input Y planes (numpy) or None.
        Upscales GPU_BATCH frames at a time into 4:2:0 planes kept in RAM while the clip's low-res consistency gate is
        accumulated; if the gate is below gate.TRIGGER_DB the luma is back-projected (gate.py); then every plane goes to the
        encoder thread. Returns (frames, gate before, gate after, alpha)."""
        torch, F = self.torch, self.torch.nn.functional
        from . import gate
        from .rrdbnet import stack_neighbors
        from .yuv import to_yuv420
        T, _, h, w = frames.shape
        s, tf = model.scale, model.tframes
        r = tf // 2
        ph, pw = (h % 2, w % 2) if s == 2 else (0, 0)
        if y_in is not None and y_in.shape[0] != T:
            log(f"input luma has {y_in.shape[0]} frames, the clip {T}: gate guard off")
            y_in = None
        yin_gpu = torch.from_numpy(y_in).to("cuda")[:, None] if y_in is not None else None
        out, sq, cnt = [], 0.0, 0
        with torch.inference_mode():
            for i in range(0, T, GPU_BATCH):
                j = min(T, i + GPU_BATCH)
                if tf > 1:
                    a, b = max(0, i - r), min(T, j + r)
                    x = stack_neighbors(frames[a:b].float().div_(255), tf)[i - a: i - a + (j - i)]
                else:
                    x = frames[i:j].float().div_(255)
                if ph or pw:
                    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = model(x.contiguous(memory_format=torch.channels_last))
                y = y.float()[..., : s * h, : s * w].clamp_(0, 1)
                Y, cb, cr = to_yuv420(y)
                if yin_gpu is not None:
                    q, c = gate.gate_sq(Y, yin_gpu[i:j], s)
                    sq, cnt = sq + q, cnt + c
                out.append([p[:, 0].to(torch.uint8).cpu().numpy() for p in (Y, cb, cr)])
            g0 = gate.psnr(sq, cnt) if cnt else float("nan")
            g1, alpha_total = g0, 0.0
            steps = 0
            # below the trigger: lift the clip to AIM_DB with partial luma back-projection steps
            while cnt and g0 < gate.TRIGGER_DB and g1 < gate.AIM_DB - 0.02 and steps < 8:
                alpha = gate.alpha_for(g1)
                sq, cnt = 0.0, 0
                for bi, i in enumerate(range(0, T, GPU_BATCH)):
                    j = min(T, i + GPU_BATCH)
                    Yc = gate.back_project(torch.from_numpy(out[bi][0]).to("cuda")[:, None], yin_gpu[i:j], s, alpha)
                    q, c = gate.gate_sq(Yc, yin_gpu[i:j], s)
                    sq, cnt = sq + q, cnt + c
                    out[bi][0] = Yc[:, 0].to(torch.uint8).cpu().numpy()
                g1, steps = gate.psnr(sq, cnt), steps + 1
                alpha_total = 1 - (1 - alpha_total) * (1 - alpha)
            alpha = alpha_total
        for planes in out:
            enc.put(planes)
        enc.put(_END)
        return T, g0, g1, alpha


# ------------------------------------------------------------------------------------------------ batch driver
def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m sn85up.main INPUT_DIR OUTPUT_DIR", file=sys.stderr)
        return 64
    in_dir, out_dir = Path(argv[0]), Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(p for p in in_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    log(f"{len(items)} items")
    if not items:
        return 0
    # probe + task in parallel (cheap), then decode ahead of the GPU
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    pool = ThreadPoolExecutor(max_workers=max(3, min(12, ncpu // 2)))
    infos = list(pool.map(lambda p: probe(str(p)), items))
    tasks = [read_task(p, i) for p, i in zip(items, infos)]
    from .router import read_frames as router_frames
    # the router's frames first (routing decides which specialists to load), then per clip its frames and luma
    rframes = [pool.submit(router_frames, str(p), i["width"], i["height"], max(i["packets"], 1)) for p, i in zip(items, infos)]
    decoded, lumas = [], []
    for p, i in zip(items, infos):
        decoded.append(pool.submit(decode, str(p), i["width"], i["height"]))
        lumas.append(pool.submit(decode_luma, str(p), i["width"], i["height"]))

    ups = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        ups = Upscaler()
        ups.get_router()
    except Exception:  # noqa: BLE001
        log("GPU path unavailable, every clip falls back to Lanczos:\n" + traceback.format_exc())

    # route every clip now and load the specialists the batch needs while nothing else competes for the CPU
    decisions = []
    for k, (path, (factor, _, _)) in enumerate(zip(items, tasks)):
        d = {"route": "natural", "probs": {"screen": float("nan")}, "uncertain": True}
        if ups is not None:
            try:
                d = ups.get_router().route(*rframes[k].result(), factor)
            except Exception:  # noqa: BLE001  a router failure must not cost the clip: use the natural specialist
                log(f"{path.name[:12]}: router failed, using the natural specialist:\n{traceback.format_exc(limit=2)}")
        decisions.append(d)
        rframes[k] = None
    if ups is not None:
        for key in sorted({(t[0], d["route"]) for t, d in zip(tasks, decisions)}):
            try:
                ups.get_model(key)
            except Exception:  # noqa: BLE001  retried (and handled) per clip
                log(f"specialist {key} failed to load:\n{traceback.format_exc(limit=2)}")

    pending = []   # (item, encoder, frames, W, H)
    reserved = 0.0  # estimated output bytes of the clips encoded so far
    failed = set()
    for k, (path, info, (factor, W, H)) in enumerate(zip(items, infos, tasks)):
        name, dst = path.name, str(out_dir / path.name)
        tag = name[:12]
        enc = None
        try:
            if ups is None:
                raise RuntimeError("no GPU path")
            if time.time() - T0 > BATCH_BUDGET_S - 120 * (len(items) - k):
                raise RuntimeError("batch time budget nearly spent")
            t = time.time()
            arr = decoded[k].result()
            t_dec = time.time() - t
            frames = torch.from_numpy(arr).pin_memory().to("cuda", non_blocking=True).permute(0, 3, 1, 2)
            if frames.shape[2] * factor != H or frames.shape[3] * factor != W:
                raise RuntimeError(f"input {frames.shape[3]}x{frames.shape[2]} x{factor} != target {W}x{H}")
            decision = decisions[k]
            t_route = time.time() - t - t_dec
            key = (factor, decision["route"])
            model = ups.get_model(key)
            t_prep = time.time() - t
            n_est = frames.shape[0] * (W * H) / (1920 * 1080)
            left = len(items) - k - 1
            crf, est = LADDER[-1][0], LADDER[-1][1] * n_est
            for c, bpf in LADDER:   # leave room for the remaining clips at the smallest rung
                if bpf * n_est <= MAX_FILE and reserved + bpf * n_est + left * 600 * LADDER[-1][1] <= MAX_TOTAL:
                    crf, est = c, bpf * n_est
                    break
            reserved += est
            enc = Encoder(dst, W, H, info["rate"], crf)
            try:
                y_in = lumas[k].result()
            except Exception:  # noqa: BLE001
                log(f"{tag}: input luma decode failed, gate guard off:\n{traceback.format_exc(limit=2)}")
                y_in = None
            n, g0, g1, alpha = ups.run(model, frames, y_in, enc)
            if info["packets"] and n != info["packets"]:
                log(f"{tag}: decoded {n} frames but the input has {info['packets']} packets")
            pending.append((path, enc, n, W, H))
            log(f"{tag} x{factor} -> {decision['route']} crf {crf} "
                f"(p_screen {decision['probs']['screen']:.3f}{', uncertain' if decision['uncertain'] else ''}): {n} frames, gate {g0:.2f} dB"
                + (f" -> {g1:.2f} dB (back-projection, total alpha {alpha:.3f})" if alpha > 0 else "") + f", {time.time() - t:.2f}s (decode wait {t_dec:.2f}, route {t_route:.2f}, upscale {time.time() - t - t_prep:.2f})")
            del frames, arr
        except Exception:  # noqa: BLE001
            log(f"{tag}: GPU path failed, Lanczos fallback:\n{traceback.format_exc(limit=3)}")
            failed.add(name)
            if enc is not None:
                enc.abort()
            try:
                lanczos(str(path), dst, W, H)
            except Exception:  # noqa: BLE001
                log(f"{tag}: Lanczos fallback failed too:\n{traceback.format_exc(limit=2)}")
        decoded[k] = lumas[k] = None   # free the decoded frames

    # wait for the encoders, verify every output, fall back where needed
    for path, enc, n, W, H in pending:
        enc.join()
        dst = out_dir / path.name
        ok = enc.err is None and dst.is_file()
        if ok:
            try:
                o = probe(str(dst))
                ok = (o["width"], o["height"], o["packets"]) == (W, H, n)
                if not ok:
                    log(f"{path.name[:12]}: output {o['width']}x{o['height']} {o['packets']} frames != {W}x{H} {n}")
            except Exception as e:  # noqa: BLE001
                ok = False
                log(f"{path.name[:12]}: output probe failed: {e!r}")
        else:
            log(f"{path.name[:12]}: encoder failed: {enc.err}")
        if ok and dst.stat().st_size > MAX_FILE:   # too big: re-encode coarser (should not happen: the ladder estimates are conservative)
            tmp = out_dir / f".tmp-{path.name}"
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(dst), "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "8", *TAGS, "-an", "-f", "mp4", str(tmp)], check=False)
            if tmp.is_file() and 0 < tmp.stat().st_size <= MAX_FILE:
                os.replace(tmp, dst)
            else:
                tmp.unlink(missing_ok=True)
                ok = False
        if not ok:
            try:
                lanczos(str(path), str(dst), W, H)
            except Exception:  # noqa: BLE001
                log(f"{path.name[:12]}: Lanczos fallback failed:\n{traceback.format_exc(limit=2)}")
    for p in out_dir.iterdir():
        if p.name.startswith("."):
            p.unlink(missing_ok=True)
        else:
            os.chmod(p, 0o644)
    total = sum(p.stat().st_size for p in out_dir.iterdir())
    frames_total = sum(n for _, _, n, _, _ in pending)
    log(f"done: {len(items) - len(failed)}/{len(items)} via the GPU path, {frames_total} frames, "
        f"outputs {total / 1e6:.1f} MB, wall {time.time() - T0:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001  never let the batch exit non-zero
        log("fatal:\n" + traceback.format_exc())
        sys.exit(0)
