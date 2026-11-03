"""Kernel only (gate_up+silu, down) on the real layer with lognormal routing: quick timing for variants."""
import os
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
t = int(os.environ.get("T", "6912"))
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=float(os.environ.get("SIGMA", "0.6")))
sid, eid, npad = kcuda.align(ids, P.E)
act = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
o = kcuda.down(act, mp, sid, eid, npad, wts)
g = P.bench(lambda: kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK, act), iters=20)
d = P.bench(lambda: kcuda.down(act, mp, sid, eid, npad, wts, o), iters=20)
print(f"T={t} variant={os.environ.get('MOEK_VARIANT', '')}: gate_up+silu {g:.2f} ms, down {d:.2f} ms")
