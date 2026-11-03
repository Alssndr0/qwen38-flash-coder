"""Kernel (packed scales, tiled layout, every schedule) vs the bf16 reference and vs Marlin, many sizes/skews."""
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mpt = kcuda.tile_layout(kcuda.with_packed(P.marlin_plain(10)))
mp = P.marlin_plain(10)
worst = 0.0
for t, sig in ((1, 0.6), (2, 0.6), (3, 0.6), (5, 0.6), (8, 0.6), (8, 1.5), (12, 0.6), (13, 0.6), (16, 0.6), (32, 0.6),
               (100, 0.6), (128, 1.0), (500, 0.6), (2000, 0.6), (6912, 0.6), (6912, 1.0), (10368, 0.6)):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t + int(sig * 10), sigma=sig)
    out = kcuda.moe(x, ids, wts, mpt)
    om = P.marlin_moe(x, ids, wts, mp, 48 if t >= 256 else 0)
    rows = list(range(0, t, max(1, t // 16)))[:16]
    ref = P.reference(x, ids, wts, w13, w2, rows)
    ek = ((out[rows].float() - ref).norm() / ref.norm()).item()
    em = ((om[rows].float() - ref).norm() / ref.norm()).item()
    dm = ((out.float() - om.float()).norm() / om.float().norm()).item()
    fin = bool(torch.isfinite(out).all())
    worst = max(worst, ek - em)
    print(f"T={t:5d} sigma {sig}: kernel vs ref {ek:.4f}, Marlin vs ref {em:.4f}, kernel vs Marlin {dm:.1e}, finite {fin}", flush=True)
print(f"worst (kernel - Marlin) error vs reference: {worst:+.4f}")
