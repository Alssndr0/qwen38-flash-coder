"""Per-stage cycle split for the first CTAs (MOEK_VARIANT=TRACE): barrier wait, weight wait, compute."""
import os
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
for rows in (16, 48, 128):
    ids, wts = P.uniform_routing(rows)
    t = ids.shape[0]
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    sid, eid, npad = kcuda.align(ids, P.E)
    for _ in range(3):
        h = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
    torch.cuda.synchronize()
    tr = kcuda.mod().trace()  # [cta, warp group, kt, 3]
    m = tr[:, :, 5:35].float().mean(dim=(0, 2))
    print(f"{rows} rows: cycles per stage (warp_m 0 / 1): barrier {m[0, 0]:.0f}/{m[1, 0]:.0f}  weights wait "
          f"{m[0, 1]:.0f}/{m[1, 1]:.0f}  compute {m[0, 2]:.0f}/{m[1, 2]:.0f}  total {m[0].sum():.0f}")
