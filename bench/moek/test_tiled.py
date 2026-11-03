"""Tiled layout == Marlin layout (bitwise), then decode sizes vs csf Marlin (graphs, L2 flushed) and prefill."""
import statistics
import torch
import kcuda
import problem as P
w13, w2 = P.real_weights(10)
mc = P.marlin_prepare(w13, w2)  # csf (MBX_MARLIN_CSF=1)
del w13, w2
mpp = P.marlin_plain(10)  # plain scales, Marlin layout: the reference path of the kernel
mpt = kcuda.tile_layout(kcuda.with_packed(mpp))
kcuda.mod()
for t in (6912, 64, 8, 1):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, sigma=0.6)
    print(f"T={t}: tiled packed == plain marlin-layout {torch.equal(kcuda.moe(x, ids, wts, mpp), kcuda.moe(x, ids, wts, mpt))}", flush=True)
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


for t in (1, 2, 4, 8, 16, 32, 64, 128):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t, sigma=0.6)
    r = compare({"m": capture(lambda: P.marlin_moe(x, ids, wts, mc, 0)), "k": capture(lambda: kcuda.moe(x, ids, wts, mpt))})
    print(f"T={t:3d}: csf Marlin {r['m']:.3f} ms, kernel tiled {r['k']:.3f} ms ({(r['k'] / r['m'] - 1) * 100:+.1f} %)", flush=True)
t = 6912
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=0.6)
print(f"T=6912 layer: csf Marlin {P.bench(lambda: P.marlin_moe(x, ids, wts, mc, 48)):.2f} ms, kernel tiled "
      f"{P.bench(lambda: kcuda.moe(x, ids, wts, mpt)):.2f} ms")
