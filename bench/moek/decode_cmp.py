"""Decode sizes under CUDA graphs, read-only L2 flush before each replay, one process (CSF=1): the serve's csf Marlin
vs the kernel on packed scales; replays alternate, medians of 200."""
import statistics
import torch
import kcuda
import problem as P

w13, w2 = P.real_weights(10)
mc = P.marlin_prepare(w13, w2)  # csf (this process runs with MBX_MARLIN_CSF=1)
mpp = kcuda.with_packed(P.marlin_plain(10))
del w13, w2
kcuda.mod()
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
        out = fn()
    return g, out


def compare(graphs, reps=200):
    t = {k: [] for k in graphs}
    for _ in range(reps):
        for k, g in graphs.items():
            SINK.copy_(FLUSH.view(torch.int64).max())
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record()
            torch.cuda.synchronize()
            t[k].append(a.elapsed_time(b))
    return {k: statistics.median(v) for k, v in t.items()}


for t in (1, 2, 4, 8, 16, 32, 64, 128):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t, sigma=0.6)
    touched = torch.unique(ids).numel()
    gm, om = capture(lambda: P.marlin_moe(x, ids, wts, mc, 0))
    gk, ok = capture(lambda: kcuda.moe(x, ids, wts, mpp))
    r = compare({"m": gm, "k": gk})
    gm.replay(); gk.replay(); torch.cuda.synchronize()
    d = ((om.float() - ok.float()).norm() / om.float().norm()).item()
    print(f"T={t:3d} experts {touched:3d}: csf Marlin {r['m']:.3f} ms, kernel {r['k']:.3f} ms ({(r['k'] / r['m'] - 1) * 100:+.1f} %), "
          f"rel diff {d:.1e}", flush=True)
