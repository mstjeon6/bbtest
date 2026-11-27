"""Dev check (needs cv2): sn85up.gate's torch Lanczos4 downscale, blur and PSNR against cv2, as the validator computes them."""
import sys
import cv2
import numpy as np
import torch
from sn85up import gate

rng = np.random.default_rng(0)
worst = 0.0
for s, (H, W) in [(4, (1080, 1920)), (4, (1920, 1080)), (2, (1080, 1920)), (2, (1920, 1080))]:
    for trial in range(3):
        yo = cv2.GaussianBlur(rng.uniform(16, 235, (H, W)).astype(np.float32), (0, 0), 1 + trial)
        yi = cv2.resize(yo, (W // s, H // s), interpolation=cv2.INTER_AREA) + rng.normal(0, 3 + 2 * trial, (H // s, W // s)).astype(np.float32)
        yo, yi = yo.round(), yi.round()
        down_cv = cv2.resize(yo, (W // s, H // s), interpolation=cv2.INTER_LANCZOS4)
        d = cv2.GaussianBlur(down_cv, (7, 7), 1.0, borderType=cv2.BORDER_REFLECT_101) - cv2.GaussianBlur(yi, (7, 7), 1.0, borderType=cv2.BORDER_REFLECT_101)
        db_cv = 10 * np.log10(255 ** 2 / np.mean(d.astype(np.float64) ** 2))
        to = torch.from_numpy(yo)[None, None]; ti = torch.from_numpy(yi)[None, None]
        down_t = gate.lanczos4_down(to, s)[0, 0].numpy()
        sq, n = gate.gate_sq(to, ti, s)
        db_t = gate.psnr(sq, n)
        md = np.abs(down_t - down_cv).max()
        worst = max(worst, abs(db_t - db_cv))
        print(f"x{s} {W}x{H} trial {trial}: max |down diff| {md:.4f}, gate cv2 {db_cv:.4f} torch {db_t:.4f} dB")
print(f"worst gate difference {worst:.4f} dB")
sys.exit(0 if worst < 0.02 else 1)
