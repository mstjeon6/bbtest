"""The validator's low-resolution consistency gate, on the GPU, and the guard that keeps every clip above it.

Validator (vidaio/scoring/upscaling_generative.py): every output frame's luma (limited-range Y plane) is downscaled to the
input size with cv2.resize(INTER_LANCZOS4), both it and the input's Y plane are Gaussian-blurred (7x7, sigma 1.0,
BORDER_REFLECT_101), and the gate is the global PSNR over all frames of the clip; a clip below 28 dB scores zero.

For an exact integer factor s, cv2's Lanczos4 downscale samples source x = s*X + (s-1)/2 with a fixed 8-tap kernel
(fraction 0.5) and replicated borders, i.e. one separable strided convolution: implemented here and checked against
cv2 (tools/check_gate.py).

Guard: if a clip's gate is below TRIGGER_DB, partial luma back-projection steps Y += alpha * up(blur(blur(Y_in - down(Y))))
(alpha = 1 - 10^((gate - aim) / 20), recomputed after each step, at most 8) lift it to AIM_DB. A clip below 28 dB
scores zero, so any correction beats failing; but it costs fidelity and temporal stability (forced 34.4 -> 35.6 dB on a
heavily degraded x4 screen clip: score 0.886 -> 0.781, DISTS +0.014, LPIPS +0.030; unblurred residual: 0.575), hence the
narrow trigger: clips at or above TRIGGER_DB are never touched.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

# The gate is 28 dB. Our measurement equals the validator's to < 0.02 dB (tools/check_gate.py; 34.39 vs 34.41 on a real
# clip), so the guard only acts on clips that would fail or nearly fail: below TRIGGER_DB it lifts them to AIM_DB.
TRIGGER_DB = float(os.environ.get("SN85UP_GATE_TRIGGER", "28.4"))   # env: test knob only (the sandbox has no env)
AIM_DB = TRIGGER_DB + 0.2
BP_BLUR = int(os.environ.get("SN85UP_BP_BLUR", "2"))
_lanczos_cache: dict = {}


def _lanczos4_taps(device) -> torch.Tensor:
    key = str(device)
    if key not in _lanczos_cache:
        f = 0.5
        xs = [f + 3 - i for i in range(8)]                     # distances of taps sx-3 .. sx+4 from the sample point
        w = []
        for x in xs:
            if abs(x) < 1e-12:
                w.append(1.0)
            else:
                w.append(math.sin(math.pi * x) * math.sin(math.pi * x / 4) / (math.pi * math.pi * x * x / 4))
        t = torch.tensor(w, dtype=torch.float32, device=device)
        _lanczos_cache[key] = t / t.sum()
    return _lanczos_cache[key]


def lanczos4_down(y: torch.Tensor, s: int) -> torch.Tensor:
    """(N, 1, H, W) float -> (N, 1, H/s, W/s): cv2.resize(INTER_LANCZOS4) for an integer factor s."""
    k = _lanczos4_taps(y.device)
    o = (s - 1) // 2                                            # sx = s*X + o for fraction 0.5 (s = 2 or 4)
    # taps at sx-3 .. sx+4: pad 3-o on the left (and enough on the right), replicate borders, stride s
    pl, pr = 3 - o, 4 + o
    x = F.pad(y, (pl, pr, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, 8), stride=(1, s))
    x = F.pad(x, (0, 0, pl, pr), mode="replicate")
    return F.conv2d(x, k.view(1, 1, 8, 1), stride=(s, 1))[..., : y.shape[-2] // s, : y.shape[-1] // s]


def _gauss7(device) -> torch.Tensor:
    g = torch.exp(-torch.arange(-3, 4, dtype=torch.float32, device=device) ** 2 / 2.0)
    return g / g.sum()


def blur(y: torch.Tensor) -> torch.Tensor:
    g = _gauss7(y.device)
    x = F.conv2d(F.pad(y, (3, 3, 0, 0), mode="reflect"), g.view(1, 1, 1, 7))
    return F.conv2d(F.pad(x, (0, 0, 3, 3), mode="reflect"), g.view(1, 1, 7, 1))


def luma_limited(rgb01: torch.Tensor) -> torch.Tensor:
    """(N, 3, h, w) full-range RGB [0, 1] -> BT.709 limited-range Y in 8-bit units, (N, 1, h, w), rounded."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return (16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)).round()


def gate_sq(y_out: torch.Tensor, y_in: torch.Tensor, s: int) -> tuple[float, int]:
    """Sum of squared blurred differences and count: y_out (N, 1, H, W) output Y, y_in (N, 1, H/s, W/s) input Y."""
    d = blur(lanczos4_down(y_out.float(), s)) - blur(y_in.float())
    return float(d.double().pow(2).sum()), d.numel()


def psnr(sq: float, n: int) -> float:
    mse = sq / max(n, 1)
    return 99.0 if mse <= 0 else min(99.0, 10.0 * math.log10(255.0 ** 2 / mse))


def alpha_for(gate_db: float, target: float = AIM_DB) -> float:
    return 0.0 if gate_db >= target else min(1.0, 1.0 - 10.0 ** ((gate_db - target) / 20.0))


def back_project(y_out: torch.Tensor, y_in: torch.Tensor, s: int, alpha: float) -> torch.Tensor:
    """One partial luma back-projection step; y_out (N, 1, H, W) 8-bit Y codes -> corrected, rounded, clamped."""
    e = y_in.float() - lanczos4_down(y_out.float(), s)
    for _ in range(BP_BLUR):   # low-pass the residual: the gate only sees blur(e); unblurred it injects input noise
        e = blur(e)
    up = F.interpolate(e, scale_factor=s, mode="bicubic", align_corners=False)
    return (y_out.float() + alpha * up).round().clamp(16, 235)
