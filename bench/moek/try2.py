"""Row-group prototype (v2) vs Marlin: correctness, per-GEMM sweep, whole layer."""
import itertools
import os
import sys

import torch

import moek
import problem as P

t = int(os.environ.get("T", "6912"))
sigma = float(os.environ.get("SIGMA", "0.6"))
w13 = P.make_weights(2 * P.INTER, P.HID, 1)
w2 = P.make_weights(P.HID, P.INTER, 2)
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=sigma)
mp = P.marlin_prepare(w13, w2)
rows = list(range(0, t, max(1, t // 24)))[:24]
ref = P.reference(x, ids, wts, w13, w2, rows)
print(f"T={t} sigma={sigma}: marlin bm48 {P.bench(lambda: P.marlin_moe(x, ids, wts, mp, 48)):.2f} ms per layer")
c0 = dict(RB=32, BN=64, BK=64, num_warps=4, num_stages=2)
out = moek.moe_rg(x, ids, wts, w13, w2, c0, c0)
err = (out[rows].float() - ref).norm() / ref.norm()
print(f"v2 rel err vs reference {err:.4f}")
if err > 0.02:
    sys.exit("wrong result")
h = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
act = torch.empty((t * P.TOPK, P.INTER), dtype=torch.bfloat16, device=P.DEV).normal_()
o = torch.empty((t * P.TOPK, P.HID), dtype=torch.bfloat16, device=P.DEV)
res = {"gate_up": [], "down": []}
rgs = {rb: moek.row_groups(ids, P.E, rb) for rb in (16, 32, 64)}
for RB, BN, BK, nw, ns in itertools.product((16, 32, 64), (32, 64, 128), (64, 128), (4, 8), (1, 2, 3)):
    if RB * BN > 64 * 128:
        continue
    cfg = dict(RB=RB, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    rg = rgs[RB]
    for name, fn in (("gate_up", lambda: moek.moe_gemm_rg(x, *w13, rg, P.TOPK, h, **cfg)),
                     ("down", lambda: moek.moe_gemm_rg(act, *w2, rg, 1, o, tw=wts.flatten(), **cfg))):
        try:
            ms = P.bench(fn, iters=5, warm=2)
        except Exception:
            continue
        res[name].append((ms, cfg))
for name in res:
    res[name].sort(key=lambda r: r[0])
    print(name, "best:")
    for ms, cfg in res[name][:4]:
        print(f"   {ms:6.2f} ms  {cfg}")
b1, b2 = res["gate_up"][0][1], res["down"][0][1]
print(f"v2 whole layer (best configs): {P.bench(lambda: moek.moe_rg(x, ids, wts, w13, w2, b1, b2)):.2f} ms per layer")
