"""Per-unit timeline of CTA 0 in the down GEMM at T=8 (MOEK_VARIANT=TRACE), cycles relative to the first stamp."""
import os
import torch
import kcuda
import problem as P
mpp = kcuda.with_packed(P.marlin_plain(10))
t = 8
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, seed=t, sigma=0.6)
kcuda._CUR["ids"] = ids.flatten()
sid, eid, npad = kcuda.align(ids, P.E)
act = kcuda.gate_up(x, mp := mpp, sid, eid, npad, P.TOPK)
FLUSH = torch.ones(64 << 20, dtype=torch.uint8, device="cuda")
for _ in range(3):
    FLUSH.view(torch.int64).max()
    o = kcuda.down(act, mpp, sid, eid, npad, wts)
torch.cuda.synchronize()
tr = kcuda.mod().trace()[0].double()
base = tr[0, 4]
print("unit: published  got   weights  done  epi   prod-last-issue   (us at 2.17 GHz, from first publish)")
for u in range(min(40, int((tr[:, 0] > 0).sum()))):
    r = (tr[u] - base) / 2170
    print(f"{u:3d}: {r[4]:7.2f} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} {r[3]:7.2f} {r[5]:7.2f}")
