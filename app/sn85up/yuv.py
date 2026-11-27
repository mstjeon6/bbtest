"""Full-range RGB on the GPU -> 8-bit BT.709 limited-range 4:2:0 planes, as ffmpeg writes them (left-sited chroma:
[1 2 1]/4 horizontally at even columns, then the mean of each vertical pair). One rounding, straight from the float
model output."""
import torch
import torch.nn.functional as F


def to_yuv420(rgb01: torch.Tensor):
    """(N, 3, H, W) RGB in [0, 1] -> Y (N, 1, H, W), Cb, Cr (N, 1, H/2, W/2), rounded 8-bit codes (float)."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Y, cb, cr = 16.0 + 219.0 * y, 128.0 + 224.0 * (b - y) / 1.8556, 128.0 + 224.0 * (r - y) / 1.5748
    k = torch.tensor([0.25, 0.5, 0.25], device=rgb01.device, dtype=rgb01.dtype).view(1, 1, 1, 3)

    def down(c):
        return F.avg_pool2d(F.conv2d(F.pad(c, (1, 1, 0, 0), mode="replicate"), k, stride=(1, 2)), (2, 1))

    return Y.round().clamp(16, 235), down(cb).round().clamp(16, 240), down(cr).round().clamp(16, 240)
