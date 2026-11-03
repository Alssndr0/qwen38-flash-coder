"""Build/load the CUDA kernel (csrc/moe_w4a16.cu) and the layer built on it."""
import os

import torch
from torch.utils.cpp_extension import load

_mod = None


def mod():
    global _mod
    if _mod is None:
        here = os.path.dirname(os.path.abspath(__file__))
        bd = os.environ.get("MOEK_BUILD", "/kit/cache/moek-build")
        os.makedirs(bd, exist_ok=True)
        var = os.environ.get("MOEK_VARIANT", "")  # e.g. "NO_MMA,NO_A": ablation builds
        defs = [f"-DMOEK_{v}" for v in var.split(",") if v]  # NAME or NAME=value
        tag = var.replace(",", "-").replace("=", "")
        bd = bd + ("-" + tag if var else "")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(here, "..", "..", "files", "mbx_moek.cu")  # the serve's kernel: one source
        _mod = load(name="moek_cuda" + ("_" + tag.replace("-", "_") if var else ""),
                    sources=[src], build_directory=bd,
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo"] + defs,
                    verbose=bool(os.environ.get("MOEK_VERBOSE")))
    return _mod


BM = int(os.environ.get("MOEK_BM_PY", "96"))


def align(ids, e, bm=None):
    bm = bm or BM
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    return moe_align_block_size(ids, bm, e, None, ignore_invalid_experts=True)


_E0 = None


def _none():
    global _E0
    if _E0 is None:
        _E0 = torch.empty(0, dtype=torch.uint8, device="cuda")
    return _E0


def _scales(mp, i):
    """(plain S, packed SP, exc_off, exc_ent) for matrix i (0 = w13, 1 = w2); mp may carry packed scales at 6/7."""
    if len(mp) > 6 and mp[6 + i] is not None:
        sp, off, ent = mp[6 + i]
        return _none(), sp, off, ent if ent.numel() else _none()
    return mp[1 + 3 * i].view(torch.uint8), _none(), _none(), _none()


def tile_layout(mp, gate_up_pairs=True):
    """Marlin weights + packed scales -> the kernel's tiled layout (same bytes, reordered):
    W [E, K/16, 2N] int32 -> per (expert, 256-column item, 64-k stage) one 8 KB block in shared-memory order
    (k16 row r: [tile t0, t0+1][tile t1, t1+1] x 512 B); SP -> per (expert, item, run) the two 640 B runs.
    Returns a new mp with entries 0/3 (weights) and 6/7 (packed scales) replaced; plain scales are dropped."""
    out = list(mp)
    for wi, si, gu in ((0, 6, True), (3, 7, False)):
        w = mp[wi]
        e, kr, n2 = w.shape
        n = n2 // 2
        nt = n // 256
        tiles = w.view(e, kr, n // 64, 128)  # [E, K/16, tiles, 128 int32]
        if gu:
            t = torch.arange(nt, device=w.device)
            idx = torch.stack([2 * t, 2 * t + 1, n // 128 + 2 * t, n // 128 + 2 * t + 1], 1)  # [nt, 4]
        else:
            idx = torch.arange(n // 64, device=w.device).view(nt, 4)
        g = tiles[:, :, idx.flatten(), :].view(e, kr // 4, 4, nt, 4, 128)  # [E, stage, k16, item, 4 tiles, 128]
        out[wi] = g.permute(0, 3, 1, 2, 4, 5).contiguous().view(e, kr, n2)
        sp, off, ent = mp[si]
        nb = sp.shape[1]
        runs = sp.view(e, nb, n // 128, RUN_BYTES_PY)
        if gu:
            ridx = torch.stack([torch.arange(nt, device=w.device), n // 256 + torch.arange(nt, device=w.device)], 1)
        else:
            ridx = torch.arange(n // 128, device=w.device).view(nt, 2)
        sp2 = runs[:, :, ridx.flatten(), :].view(e, nb, nt, 2 * RUN_BYTES_PY).permute(0, 2, 1, 3).contiguous()
        out[si] = (sp2, off, ent)
    out[1] = out[4] = None
    return tuple(out)


RUN_BYTES_PY = 16 + 8 * 64
TILED = int(os.environ.get("MOEK_TILED", "1"))


def with_packed(mp):
    """mp + packed scales (the serve's format, mbx_moek.pack_scales) for both matrices."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "files"))
    import mbx_moek as M
    return tuple(mp[:6]) + (M.pack_scales(mp[1], True), M.pack_scales(mp[4], False))


_BUF = {}


def _streamk():
    """Stream-k workspace (2 slots per SM x 96 rows x 256 fp32) and per-item counters (kept zero by the kernel)."""
    if "ws" not in _BUF:
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        _BUF["ws"] = torch.empty(2 * sms * BM * 256, dtype=torch.float32, device="cuda")
        _BUF["cnt"] = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    return _BUF["ws"], _BUF["cnt"]


STREAMK_MAX_T = int(os.environ.get("MOEK_STREAMK_MAX_T", "0"))  # stream-k up to this many tokens (off: slower)
SPLIT_MAX_T = int(os.environ.get("MOEK_SPLIT_MAX_T", "12"))  # split mode up to this many tokens
SPLIT_UNITS = int(os.environ.get("MOEK_SPLIT_UNITS", str(20 * 48)))  # split until at least this many units
ORDER = int(os.environ.get("MOEK_ORDER", "0"))  # mode 2 unit order: 0 (tile, k), 1 (k, tile)
NS_GATE = int(os.environ.get("MOEK_NS_GATE", "5"))  # split mode: gate_up in 5 k units (8 stages), measured best
NS_DOWN = int(os.environ.get("MOEK_NS_DOWN", "1"))  # down (10 stages) unsplit: shorter units lose more than balance gains


def _mode(t, n, k, e, num_ids, sid_len, force=0):
    """(mode, nsplit, partials) for t tokens."""
    if t <= SPLIT_MAX_T:
        ktiles = k // 64
        blocks = min(sid_len // BM, (num_ids + BM - 1) // BM + min(e, num_ids))
        items = (n // 256) * blocks
        ns = 1
        for d in range(1, ktiles + 1):
            if ktiles % d or (ktiles // d) % 2:
                continue
            ns = d
            if items * d >= SPLIT_UNITS:
                break
        ns = force or ns
        need = ns * num_ids * n
        pt = _BUF.get("part")
        if pt is None or pt.numel() < need:
            # sized once for the largest split call (graphs keep the address): K/128 units x 16 tokens x top-5 x N
            pt = _BUF["part"] = torch.empty(max(need, (k // 128) * SPLIT_MAX_T * 5 * max(n, 2560)), dtype=torch.float32,
                                            device="cuda")
        return 2, ns, pt
    return (1 if t <= STREAMK_MAX_T else 0), 1, _none()


def _tiled(mp):
    return len(mp) > 6 and mp[1] is None


def gate_up(x, mp, sid, eid, npad, top_k, out=None):
    """silu(x @ gate) * (x @ up) per (token, expert) row: [T * top_k, N/2] (fused silu_and_mul)."""
    w1, g1 = mp[0], mp[2]
    n = w1.shape[2] // 2
    out = torch.empty((x.shape[0] * top_k, n // 2), dtype=torch.bfloat16, device=x.device) if out is None else out
    s, sp, off, ent = _scales(mp, 0)
    mode, ns, pt = _mode(x.shape[0], n, x.shape[1], w1.shape[0], out.shape[0], sid.shape[0], NS_GATE)
    mod().moe_gemm(x, w1, s, g1, sid, eid, npad, g1, out, top_k, True, sp, off, ent, *_streamk(), mode, ns, pt, ORDER, _tiled(mp))
    if mode == 2:
        mod().moe_split_epilogue(pt, ns, _CUR["ids"], g1, g1, out, True, n)
    return out


def down(act, mp, sid, eid, npad, wts, out=None):
    w2, g2 = mp[3], mp[5]
    n = w2.shape[2] // 2
    out = torch.empty((act.shape[0], n), dtype=torch.bfloat16, device=act.device) if out is None else out
    s, sp, off, ent = _scales(mp, 1)
    mode, ns, pt = _mode(wts.shape[0], n, act.shape[1], w2.shape[0], out.shape[0], sid.shape[0], NS_DOWN)
    mod().moe_gemm(act, w2, s, g2, sid, eid, npad, wts.flatten(), out, 1, False, sp, off, ent, *_streamk(), mode, ns, pt, ORDER, _tiled(mp))
    if mode == 2:
        mod().moe_split_epilogue(pt, ns, _CUR["ids"], g2, wts.flatten(), out, False, n)
    return out


_CUR = {}


def moe(x, ids, wts, mp, e=512):
    t, k = ids.shape
    _CUR["ids"] = ids.flatten() if ids.dtype == torch.int32 else ids.flatten().to(torch.int32)
    sid, eid, npad = align(ids, e)
    act = gate_up(x, mp, sid, eid, npad, k)
    o = down(act, mp, sid, eid, npad, wts)
    return o.view(t, k, -1).sum(1)
