"""Tiled layout == Marlin layout (bitwise), then decode sizes vs csf Marlin (graphs, L2 flushed) and prefill."""
import statistics
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mc = P.marlin_prepare(w13, w2)  # csf (MBX_MARLIN_CSF=1)
del w13, w2
mpp = kcuda.with_packed(P.marlin_plain(10))
mpt = kcuda.tile_layout(mpp)
kcuda.mod()
for t in (6912, 64, 8, 1):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, sigma=0.6)
    print(f"T={t}: tiled == marlin layout {torch.equal(kcuda.moe(x, ids, wts, mpp), kcuda.moe(x, ids, wts, mpt))}", flush=True)
del mpp
FLUSH = torch.ones(64 << 20, dtype=torch.uint8, device="cuda")
SINK = torch.empty((), dtype=torch.int64, device="cuda")


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def compare(graphs, reps=200):
    tm = {k: [] for k in graphs}
    for _ in range(reps):
        for k, g in graphs.items():
            SINK.copy_(FLUSH.view(torch.int64).max())
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record()
            torch.cuda.synchronize()
            tm[k].append(a.elapsed_time(b))
    return {k: statistics.median(v) for k, v in tm.items()}


import os
CFG = [("mode0", 0, 0, 0), ("split g5 d1", 16, 5, 1), ("split g4 d1", 16, 4, 1), ("split g10 d1", 16, 10, 1),
       ("split g2 d1", 16, 2, 1), ("split g5 d5", 16, 5, 5), ("split g20 d1", 16, 20, 1)]
for t in [int(v) for v in os.environ.get("TS", "2,4,8,16").split(",")]:
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t, sigma=0.6)
    graphs = {"csf Marlin": capture(lambda: P.marlin_moe(x, ids, wts, mc, 0))}
    for name, smt, ng, nd in CFG:
        kcuda.SPLIT_MAX_T, kcuda.NS_GATE, kcuda.NS_DOWN = smt, ng, nd
        graphs[name] = capture(lambda: kcuda.moe(x, ids, wts, mpt))
    r = compare(graphs, reps=150)
    m = r["csf Marlin"]
    print(f"T={t:3d}: csf Marlin {m:.3f} ms; " + ", ".join(f"{k} {(v / m - 1) * 100:+.1f}%" for k, v in r.items() if k != "csf Marlin"), flush=True)
