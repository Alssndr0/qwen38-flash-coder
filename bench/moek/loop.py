import os, time, torch, kcuda, problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
ids, wts = P.uniform_routing(int(os.environ.get("UNIFORM", "48")))
t = ids.shape[0]
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
sid, eid, npad = kcuda.align(ids, P.E)
h = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
torch.cuda.synchronize()
print("START", flush=True)
t0 = time.time(); n = 0
while time.time() - t0 < 12:
    for _ in range(20):
        kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK, h)
    torch.cuda.synchronize(); n += 20
print(f"{(time.time() - t0) / n * 1e3:.2f} ms per call", flush=True)
