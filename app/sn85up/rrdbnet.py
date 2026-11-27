"""Real-ESRGAN generator (RRDBNet: 23 RRDB blocks, 64 features, 16.7M parameters), x2 and x4.

Same architecture and state-dict keys as xinntao/Real-ESRGAN (BSD-3-Clause). Our fine-tuned checkpoints may read
several LR frames (tframes = 3: the channel stack [prev, cur, next], edge frames replicated). x2 checkpoints
pixel-unshuffle the LR frame by 2 first (12 channels per frame at half size).

Weights are .safetensors files, read by a dependency-free reader (the format is an 8-byte little-endian header
length, a JSON header, then the raw tensor bytes).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def read_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """(state dict, metadata) of a .safetensors file (float tensors only)."""
    data = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + n])
    meta = header.pop("__metadata__", {}) or {}
    base = 8 + n
    sd = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        t = torch.frombuffer(bytearray(data[base + a: base + b]), dtype=_DTYPES[info["dtype"]])
        sd[name] = t.reshape(info["shape"]) if info["shape"] else t.reshape(())
    return sd, meta


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale: int = 4, nf: int = 64, nb: int = 23, gc: int = 32, tframes: int = 1):
        super().__init__()
        assert scale in (2, 4) and tframes % 2 == 1
        self.scale, self.tframes = scale, tframes
        self.conv_first = nn.Conv2d(3 * tframes * (4 if scale == 2 else 1), nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """x (N, 3*tframes, h, w) RGB in [0, 1] -> (N, 3, scale*h, scale*w). For x2, h and w must be even."""
        feat = self.conv_first(F.pixel_unshuffle(x, 2) if self.scale == 2 else x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def stack_neighbors(seq: torch.Tensor, tframes: int, ctx: int = 0) -> torch.Tensor:
    """seq (L, 3, h, w) consecutive frames -> (L - 2*ctx, 3*tframes, h, w): the stack [t-r .. t+r] for every frame t
    that has ctx context frames on each side; beyond the sequence the edge frame is replicated."""
    r = tframes // 2
    if r == 0:
        return seq
    L = seq.shape[0]
    idx = torch.arange(ctx, L - ctx, device=seq.device)
    return torch.cat([seq.index_select(0, (idx + d).clamp(0, L - 1)) for d in range(-r, r + 1)], 1)


def load_model(path: str | Path, device="cuda") -> RRDBNet:
    sd, meta = read_safetensors(path)
    cin = sd["conv_first.weight"].shape[1]
    tf = int(meta.get("tframes", 1))
    scale = 2 if cin == 12 * tf else 4
    if cin != 3 * tf * (4 if scale == 2 else 1):
        raise ValueError(f"{path}: conv_first has {cin} input channels, inconsistent with tframes={tf}")
    m = RRDBNet(scale, tframes=tf)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return m.to(device).eval()
