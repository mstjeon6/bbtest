"""Content router: decides which specialist upscales a clip (screen content vs other = AI video, 3D renders, camera footage).

router_v2 is a 4-class logistic regression (ai / camera / render / screen) on OpenAI CLIP RN50 image embeddings; the decision
uses p(screen) only: screen if p(screen) >= 0.5. It was trained on two independent datasets together (synthetic and real
screen recordings, Blender renders, Wan 2.2 video, camera footage; 6330 clips) and makes no screen/other error on the
held-out clips of either (933 clips) nor on the ten practice inputs.
Features:
  frames  3 pairs of consecutive frames at 20 / 50 / 80 % of the clip, decoded as raw yuv420p and converted to RGB with the
          BT.709 limited-range matrix, chroma upsampled bilinearly (half-pixel centres);
  x2      inputs are area-downscaled by 2 first (every clip is measured at the x4 input scale);
  CLIP    short side -> 224 (bicubic, antialiased), 3 crops of 224 along the long side, L2-normalised, mean over the crops,
          re-normalised: one 1024-d vector per frame; the clip vector is the re-normalised mean of the 6 frame vectors;
  head    standardised with the stored mean / std, logits = x @ W + b, softmax.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .clip_rn50 import load_visual

THRESHOLD = 0.5
UNCERTAIN = (0.2, 0.8)
POS = (0.2, 0.5, 0.8)
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def frame_indices(n: int) -> list[int]:
    out = []
    for p in POS:
        i = min(max(int(round(p * (n - 1))), 0), max(n - 2, 0))
        out += [i, min(i + 1, n - 1)]
    return out


def read_frames(path: str, w: int, h: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The router's 6 frames of `path` as yuv420p planes: Y (6, h, w), UV (6, 2, h/2, w/2) uint8."""
    idx = frame_indices(n)
    uniq = sorted(set(idx))
    sel = "+".join(f"eq(n\\,{i})" for i in uniq)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", path, "-vf", f"select={sel},format=yuv420p",
                          "-fps_mode", "passthrough", "-frames:v", str(len(uniq)), "-f", "rawvideo", "-"],
                         check=True, capture_output=True, timeout=300).stdout
    fs = w * h * 3 // 2
    k = len(raw) // fs
    if k == 0:
        raise RuntimeError("router: decoded no frames")
    a = np.frombuffer(raw[: k * fs], np.uint8).reshape(k, fs)
    Y = a[:, : w * h].reshape(k, h, w)
    UV = a[:, w * h:].reshape(k, 2, h // 2, w // 2)
    pos = {f: min(j, k - 1) for j, f in enumerate(uniq)}
    return Y[[pos[i] for i in idx]], UV[[pos[i] for i in idx]]


def yuv420_to_rgb(Y: torch.Tensor, UV: torch.Tensor) -> torch.Tensor:
    """BT.709 limited-range 4:2:0 (uint8) -> full-range RGB in [0, 1] (unclamped), chroma upsampled bilinearly."""
    y = Y.float() - 16.0
    u = F.interpolate(UV[:, 0:1].float() - 128.0, scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
    v = F.interpolate(UV[:, 1:2].float() - 128.0, scale_factor=2, mode="bilinear", align_corners=False)[:, 0]
    u, v = u[..., : y.shape[-2], : y.shape[-1]], v[..., : y.shape[-2], : y.shape[-1]]
    ys, cb, cr = y * (255 / 219), u * (255 / 224), v * (255 / 224)
    return torch.stack([ys + 1.5748 * cr, ys - 0.18733 * cb - 0.46813 * cr, ys + 1.8556 * cb], 1) / 255


class Router:
    def __init__(self, weights_dir: str | Path, device: str = "cuda"):
        weights_dir = Path(weights_dir)
        self.device = device
        self.clip = load_visual(str(weights_dir / "clip_rn50_visual.pt"), device)
        d = torch.load(weights_dir / "router_v2.pt", map_location="cpu", weights_only=True)
        assert d["features"] == "clip" and "screen" in d["classes"], d["classes"]
        self.classes = list(d["classes"])
        self.W, self.b = d["W"].float().to(device), d["b"].float().to(device)
        self.mu, self.sd = d["mean"].float().to(device), d["std"].float().to(device)
        self.clip_mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed(self, iy: np.ndarray, iuv: np.ndarray, scale: int) -> torch.Tensor:
        """6 frames as yuv420p planes -> the clip's (1024,) embedding."""
        rgb = yuv420_to_rgb(torch.from_numpy(np.ascontiguousarray(iy)).to(self.device),
                            torch.from_numpy(np.ascontiguousarray(iuv)).to(self.device)).float().clamp(0, 1)
        if scale == 2:
            rgb = F.interpolate(rgb, scale_factor=0.5, mode="area")
        n, _, h, w = rgb.shape
        s = 224 / min(h, w)
        H, W = max(224, round(h * s)), max(224, round(w * s))
        r = F.interpolate(rgb, size=(H, W), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        L = max(H, W)
        offs = [0, (L - 224) // 2, L - 224]
        crops = torch.cat([r[:, :, o:o + 224, :224] if H > W else r[:, :, :224, o:o + 224] for o in offs])
        e = self.clip((crops - self.clip_mean) / self.clip_std, pos_embedding=True).float()
        e = e / e.norm(dim=-1, keepdim=True)
        emb = e.view(3, n, -1).mean(0)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        c = emb.mean(0)
        return c / c.norm()

    @torch.inference_mode()
    def route(self, iy: np.ndarray, iuv: np.ndarray, scale: int) -> dict:
        e = self.embed(iy, iuv, scale)
        p = torch.softmax(((e - self.mu) / self.sd) @ self.W + self.b, 0).cpu().numpy()
        probs = {c: float(v) for c, v in zip(self.classes, p)}
        ps = probs["screen"]
        return {"route": "screen" if ps >= THRESHOLD else "natural", "probs": probs,
                "uncertain": UNCERTAIN[0] < ps < UNCERTAIN[1]}
