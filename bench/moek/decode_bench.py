"""Decode-size batches: whole MoE (kernel path vs Marlin path) on the real layer."""
import os
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
for t in (1, 4, 8, 16, 24, 32, 64, 128, 256):
    ids, wts = P.routing(t, sigma=0.6)
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    touched = torch.unique(ids).numel()
    m = P.bench(lambda: P.marlin_moe(x, ids, wts, mp, 0), iters=50)
    k = P.bench(lambda: kcuda.moe(x, ids, wts, mp), iters=50)
    fl = touched / P.E * P.floor_ms(t)["weights_ms"]
    print(f"T={t:4d} experts {touched:3d}: Marlin {m:.3f} ms, kernel {k:.3f} ms, weight floor {fl:.3f}")
