"""Decode-size MoE under CUDA graphs (as the serve runs decode): Marlin path (CSF env decides csf or plain) vs the
kernel with packed scales (needs CSF=0 for the plain scales it packs)."""
import os
import torch
import kcuda
import problem as P

csf = os.environ.get("MBX_MARLIN_CSF") == "1"
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
mpp = None if csf else kcuda.with_packed(mp)
if mpp is not None:
    kcuda.mod()


FLUSH = torch.ones(64 << 20, dtype=torch.uint8, device="cuda")
SINK = torch.empty((), dtype=torch.int64, device="cuda")


def graph_ms(fn, reps=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(reps)]
    for a, b in ev:
        SINK.copy_(FLUSH.view(torch.int64).max())  # read-only L2 flush (clean lines; the serve runs 47 other layers in between)
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return sum(a.elapsed_time(b) for a, b in ev) / reps


for t in (1, 2, 4, 8, 16, 32, 64):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t, sigma=0.6)
    touched = torch.unique(ids).numel()
    fl = touched / P.E * P.E * (2 * P.INTER * P.HID + P.HID * P.INTER) * 0.536 / 230e9 * 1e3 / P.E * P.E
    line = f"T={t:3d} experts {touched:3d} (csf-byte floor {touched * (3 * P.INTER * P.HID) * 0.536 / 230e9 * 1e3:.3f} ms):"
    line += f" Marlin{'-csf' if csf else '-plain'} {graph_ms(lambda: P.marlin_moe(x, ids, wts, mp, 0)):.3f} ms"
    if mpp is not None:
        line += f", kernel-packed {graph_ms(lambda: kcuda.moe(x, ids, wts, mpp)):.3f} ms"
    print(line, flush=True)
