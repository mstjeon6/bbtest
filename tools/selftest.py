"""Build-time self-test (CPU, no GPU at build): every specialist loads and upscales a tiny frame to the right size,
and the router loads and routes. Exits non-zero (fails the build) on any problem."""
import sys

import torch

from sn85up.main import SPECIALISTS, WEIGHTS
from sn85up.router import Router
from sn85up.rrdbnet import load_model, stack_neighbors

torch.manual_seed(0)
for (factor, route), fname in SPECIALISTS.items():
    m = load_model(WEIGHTS / fname, "cpu")
    assert m.scale == factor, f"{fname}: scale {m.scale} != {factor}"
    x = stack_neighbors(torch.rand(3, 3, 16, 24), m.tframes)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (3, 3, 16 * factor, 24 * factor), (fname, tuple(y.shape))
    assert torch.isfinite(y).all(), fname
    print(f"ok  {fname}: x{factor} {route}, tframes {m.tframes}")
r = Router(WEIGHTS, "cpu")
out = r.route((torch.rand(6, 270, 480) * 255).to(torch.uint8).numpy(), (torch.rand(6, 2, 135, 240) * 255).to(torch.uint8).numpy(), 4)
assert out["route"] in ("screen", "natural") and abs(sum(out["probs"].values()) - 1) < 1e-3, out
print(f"ok  router: {out}")
print(f"torch {torch.__version__}, python {sys.version.split()[0]}")
