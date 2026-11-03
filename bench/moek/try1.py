"""Prototype vs Marlin on one layer: correctness against the bf16 reference, then per-GEMM and whole-layer times."""
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
cnt = torch.bincount(ids.flatten().long(), minlength=P.E)
print(f"T={t} sigma={sigma}: rows per expert mean {cnt.float().mean():.1f} max {int(cnt.max())}; {P.floor_ms(t)}")
mp = P.marlin_prepare(w13, w2)
rows = list(range(0, t, max(1, t // 24)))[:24]
ref = P.reference(x, ids, wts, w13, w2, rows)
for bm in (0, 48):
    ms = P.bench(lambda: P.marlin_moe(x, ids, wts, mp, bm))
    print(f"marlin bm={bm or 'vllm'}: {ms:.2f} ms per layer")

base = dict(BM=64, BN=128, BK=128, num_warps=8, num_stages=3)
out = moek.moe(x, ids, wts, w13, w2, base, base)
err = (out[rows].float() - ref).norm() / ref.norm()
print(f"prototype rel err vs reference {err:.4f} (marlin's is ~0.006)")
if err > 0.02:
    sys.exit("wrong result")

# per-GEMM sweep (gate_up: N 1280, K 2560; down: N 2560, K 640)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402

h = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
act = torch.empty((t * P.TOPK, P.INTER), dtype=torch.bfloat16, device=P.DEV).normal_()
o = torch.empty((t * P.TOPK, P.HID), dtype=torch.bfloat16, device=P.DEV)
res = {"gate_up": [], "down": []}
for BM, BN, BK, nw, ns in itertools.product((32, 64, 128), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4)):
    if BM * BN > 128 * 128 or (BN == 256 and BK == 128 and ns > 3):
        continue
    sid, eid, npad = moe_align_block_size(ids, BM, P.E, None, ignore_invalid_experts=True)
    cfg = dict(BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    for name, fn in (("gate_up", lambda: moek.moe_gemm(x, *w13, sid, eid, npad, P.TOPK, h, **cfg)),
                     ("down", lambda: moek.moe_gemm(act, *w2, sid, eid, npad, 1, o, tw=wts.flatten(), **cfg))):
        try:
            ms = P.bench(fn, iters=5, warm=2)
        except Exception as ex:  # out of shared memory etc.
            continue
        res[name].append((ms, cfg))
for name in res:
    res[name].sort(key=lambda r: r[0])
    print(name, "best:")
    for ms, cfg in res[name][:5]:
        print(f"   {ms:6.2f} ms  {cfg}")
b1, b2 = res["gate_up"][0][1], res["down"][0][1]
ms = P.bench(lambda: moek.moe(x, ids, wts, w13, w2, b1, b2))
print(f"prototype whole layer (best configs): {ms:.2f} ms per layer")
