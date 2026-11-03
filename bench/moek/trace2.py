import torch, kcuda, problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
for rows in (16, 48):
    ids, wts = P.uniform_routing(rows)
    t = ids.shape[0]
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    sid, eid, npad = kcuda.align(ids, P.E)
    for _ in range(3):
        h = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
    torch.cuda.synchronize()
    tr = kcuda.mod().trace()
    for cta in (0, 3):
        w = tr[cta, 0]
        print(f"{rows} rows cta {cta} warp0 weights-wait per kt:", w[:40, 1].tolist())
        print(f"{rows} rows cta {cta} warp0 compute per kt     :", w[:40, 2].tolist())
