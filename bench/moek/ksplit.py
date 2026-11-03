"""Per-kernel GPU time at a decode size (torch profiler, L2 flushed before each layer): csf Marlin vs kernel modes."""
import os
import torch
import kcuda
import problem as P

t = int(os.environ.get("T", "8"))
w13, w2 = P.real_weights(10)
mc = P.marlin_prepare(w13, w2)
mpp = kcuda.with_packed(P.marlin_plain(10))
del w13, w2
kcuda.mod()
FLUSH = torch.ones(64 << 20, dtype=torch.uint8, device="cuda")
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, seed=t, sigma=0.6)


def prof(name, fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
        for _ in range(reps):
            FLUSH.view(torch.int64).max()
            fn()
        torch.cuda.synchronize()
    rows = []
    for e in pr.key_averages():
        us = getattr(e, "self_device_time_total", 0) / reps
        if us > 0.5 and "max" not in e.key.lower() and "reduce_kernel<512" not in e.key:
            rows.append((us, e.key[:70]))
    tot = sum(r[0] for r in rows)
    print(f"{name}: kernels sum {tot:.1f} us")
    for us, k in sorted(rows, reverse=True):
        print(f"   {us:7.1f} us  {k}")


prof("csf Marlin", lambda: P.marlin_moe(x, ids, wts, mc, 0))
kcuda.SPLIT_MAX_T = 0
prof("kernel mode 0", lambda: kcuda.moe(x, ids, wts, mpp))
kcuda.SPLIT_MAX_T = 16
for order in (0,):
    for ng, nd in ((5, 1), (4, 1)):
        kcuda.ORDER, kcuda.NS_GATE, kcuda.NS_DOWN = order, ng, nd
        prof(f"mode 2 order {order} ns gate {ng} down {nd}", lambda: kcuda.moe(x, ids, wts, mpp))
