"""Per-kernel time of the serve's Marlin MoE path at T=6912 (torch profiler), plain or csf scales."""
import os
import torch
import problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
t = 6912
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=0.6)
for _ in range(3):
    P.marlin_moe(x, ids, wts, mp, 48)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        P.marlin_moe(x, ids, wts, mp, 48)
    torch.cuda.synchronize()
tot = 0
rows = []
for e in prof.key_averages():
    if e.device_type == torch.autograd.DeviceType.CUDA or getattr(e, "self_device_time_total", 0) > 0:
        us = getattr(e, "self_device_time_total", 0) / 10
        if us > 1:
            rows.append((us, e.key[:90]))
for us, k in sorted(rows, reverse=True):
    tot += us
    print(f"{us / 1000:7.3f} ms  {k}")
print(f"total {tot / 1000:.3f} ms per layer (csf={os.environ.get('MBX_MARLIN_CSF')})")
