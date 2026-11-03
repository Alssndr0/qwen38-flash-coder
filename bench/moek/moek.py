"""Prototype W4A16 grouped GEMM for the routed experts (Triton), weights in the checkpoint's NVFP4 layout.

Same math as Marlin: e2m1 x e4m3 block scale is exact in bf16, bf16 x bf16 tensor-core products, fp32 sums, the
per-expert global scale (and the router weight, for the down projection) applied to the fp32 sum. Rows come in
vLLM's moe_align_block_size order (each expert's rows padded to a multiple of BM; sentinel rows are skipped).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _w4a16_moe(A, W, S, G, C, SID, EID, NPAD, TW,
               N, K, num_valid,
               stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_cm,
               TOP_K: tl.constexpr, MUL_W: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    mb = pid // num_n          # n-tiles of one row block are neighbours: the A rows stay in L2
    nb = pid % num_n
    if mb * BM >= tl.load(NPAD):
        return
    e = tl.load(EID + mb).to(tl.int64)
    if e < 0:
        return
    offs_m = mb * BM + tl.arange(0, BM)
    sid = tl.load(SID + offs_m)
    mvalid = sid < num_valid
    arow = (sid // TOP_K).to(tl.int64)
    offs_n = (nb * BN + tl.arange(0, BN)).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_kh = tl.arange(0, BK // 2)
    offs_g = tl.arange(0, BK // 16)
    a_ptr = A + arow[:, None] * stride_am + offs_k[None, :]
    w_ptr = W + e * stride_we + offs_n[:, None] * stride_wn + offs_kh[None, :]
    s_ptr = S + e * stride_se + offs_n[:, None] * stride_sn + offs_g[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptr, mask=mvalid[:, None], other=0.0)
        q = tl.load(w_ptr)                                   # [BN, BK/2] two e2m1 per byte
        sc = tl.load(s_ptr).to(tl.float8e4nv, bitcast=True).to(tl.float32)   # [BN, BK/16]
        code = tl.join(q & 0xF, q >> 4).reshape(BN, BK)      # element 2i is the low nibble
        mag = code & 7
        ex = mag >> 1
        m1 = mag & 1
        two_v = tl.where(ex == 0, m1, (2 + m1) << (ex - 1)).to(tl.float32)   # 2 * |value|
        v = tl.where((code & 8) != 0, -0.5 * two_v, 0.5 * two_v)
        w = (v.reshape(BN, BK // 16, 16) * sc[:, :, None]).reshape(BN, BK).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptr += BK
        w_ptr += BK // 2
        s_ptr += BK // 16
    acc = acc * tl.load(G + e)
    if MUL_W:
        acc = acc * tl.load(TW + sid, mask=mvalid, other=0.0)[:, None]
    c_ptr = C + sid.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptr, acc.to(tl.bfloat16), mask=mvalid[:, None])


def moe_gemm(a, w, s, g, sid, eid, npad, top_k, out, tw=None, BM=64, BN=128, BK=128, num_warps=8, num_stages=3):
    """out[sid] = a[sid // top_k] @ W[expert]^T (bf16), rows grouped per expert by moe_align_block_size(BM)."""
    n, k = w.shape[1], w.shape[2] * 2
    grid = (triton.cdiv(sid.numel(), BM) * triton.cdiv(n, BN),)
    s8 = s.view(torch.uint8)
    _w4a16_moe[grid](a, w, s8, g, out, sid, eid, npad, tw if tw is not None else g,
                     n, k, out.shape[0],
                     a.stride(0), w.stride(0), w.stride(1), s8.stride(0), s8.stride(1), out.stride(0),
                     TOP_K=top_k, MUL_W=tw is not None, BM=BM, BN=BN, BK=BK,
                     num_warps=num_warps, num_stages=num_stages)
    return out


def moe(x, ids, wts, w13, w2, cfg1: dict, cfg2: dict):
    """Whole routed-expert layer: gate_up GEMM, SiLU-and-mul, down GEMM (router weight), sum over the top-k."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    import vllm._custom_ops as ops  # noqa: F401
    t, topk = ids.shape
    e = w13[0].shape[0]
    inter = w2[0].shape[2] * 2
    bm = cfg1["BM"]
    sid, eid, npad = moe_align_block_size(ids, bm, e, None, ignore_invalid_experts=True)
    h = torch.empty((t * topk, 2 * inter), dtype=torch.bfloat16, device=x.device)
    moe_gemm(x, *w13, sid, eid, npad, topk, h, **cfg1)
    act = torch.empty((t * topk, inter), dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(act, h)
    if cfg2["BM"] != bm:
        sid, eid, npad = moe_align_block_size(ids, cfg2["BM"], e, None, ignore_invalid_experts=True)
    o = torch.empty((t * topk, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    moe_gemm(act, *w2, sid, eid, npad, 1, o, tw=wts.flatten(), **cfg2)
    return o.view(t, topk, -1).sum(1)


# ---- v2: row groups. A program owns up to 4 x RB rows of one expert and one BN-wide column tile; each weight tile
# is decoded once per k-step and used for every row chunk that has rows (empty chunks are skipped).
@triton.jit
def _dot_chunk(A, sid_ptr, c: tl.constexpr, base, nrows, a_off_k, num_valid, stride_am, w, acc,
               TOP_K: tl.constexpr, RB: tl.constexpr, BK: tl.constexpr):
    if c * RB < nrows:
        offs = base + c * RB + tl.arange(0, RB)
        sid = tl.load(sid_ptr + offs)
        ok = sid < num_valid
        arow = (sid // TOP_K).to(tl.int64)
        a = tl.load(A + arow[:, None] * stride_am + a_off_k[None, :], mask=ok[:, None], other=0.0)
        acc = tl.dot(a, w, acc)
    return acc


@triton.jit
def _store_chunk(C, sid_ptr, c: tl.constexpr, base, nrows, num_valid, stride_cm, offs_n, acc, g, TW,
                 MUL_W: tl.constexpr, RB: tl.constexpr):
    if c * RB < nrows:
        offs = base + c * RB + tl.arange(0, RB)
        sid = tl.load(sid_ptr + offs)
        ok = sid < num_valid
        acc = acc * g
        if MUL_W:
            acc = acc * tl.load(TW + sid, mask=ok, other=0.0)[:, None]
        tl.store(C + sid.to(tl.int64)[:, None] * stride_cm + offs_n[None, :], acc.to(tl.bfloat16), mask=ok[:, None])


@triton.jit
def _w4a16_moe_rg(A, W, S, G, C, SID, GE, GB, GN, NG, TW,
                  N, K, num_valid,
                  stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_cm,
                  TOP_K: tl.constexpr, MUL_W: tl.constexpr,
                  RB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    gi = pid // num_n
    nb = pid % num_n
    if gi >= tl.load(NG):
        return
    e = tl.load(GE + gi).to(tl.int64)
    base = tl.load(GB + gi)          # first row (index into SID) of the group
    nrows = tl.load(GN + gi)         # rows in the group (padded to RB), <= 4 * RB
    offs_n = (nb * BN + tl.arange(0, BN)).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_kh = tl.arange(0, BK // 2)
    offs_g = tl.arange(0, BK // 16)
    w_ptr = W + e * stride_we + offs_n[:, None] * stride_wn + offs_kh[None, :]
    s_ptr = S + e * stride_se + offs_n[:, None] * stride_sn + offs_g[None, :]
    acc0 = tl.zeros((RB, BN), dtype=tl.float32)
    acc1 = tl.zeros((RB, BN), dtype=tl.float32)
    acc2 = tl.zeros((RB, BN), dtype=tl.float32)
    acc3 = tl.zeros((RB, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        q = tl.load(w_ptr)
        sc = tl.load(s_ptr).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        code = tl.join(q & 0xF, q >> 4).reshape(BN, BK)
        mag = code & 7
        ex = mag >> 1
        m1 = mag & 1
        two_v = tl.where(ex == 0, m1, (2 + m1) << (ex - 1)).to(tl.float32)
        v = tl.where((code & 8) != 0, -0.5 * two_v, 0.5 * two_v)
        wt = tl.trans((v.reshape(BN, BK // 16, 16) * sc[:, :, None]).reshape(BN, BK).to(tl.bfloat16))
        a_off_k = k0 + offs_k
        acc0 = _dot_chunk(A, SID, 0, base, nrows, a_off_k, num_valid, stride_am, wt, acc0, TOP_K, RB, BK)
        acc1 = _dot_chunk(A, SID, 1, base, nrows, a_off_k, num_valid, stride_am, wt, acc1, TOP_K, RB, BK)
        acc2 = _dot_chunk(A, SID, 2, base, nrows, a_off_k, num_valid, stride_am, wt, acc2, TOP_K, RB, BK)
        acc3 = _dot_chunk(A, SID, 3, base, nrows, a_off_k, num_valid, stride_am, wt, acc3, TOP_K, RB, BK)
        w_ptr += BK // 2
        s_ptr += BK // 16
    g = tl.load(G + e)
    _store_chunk(C, SID, 0, base, nrows, num_valid, stride_cm, offs_n, acc0, g, TW, MUL_W, RB)
    _store_chunk(C, SID, 1, base, nrows, num_valid, stride_cm, offs_n, acc1, g, TW, MUL_W, RB)
    _store_chunk(C, SID, 2, base, nrows, num_valid, stride_cm, offs_n, acc2, g, TW, MUL_W, RB)
    _store_chunk(C, SID, 3, base, nrows, num_valid, stride_cm, offs_n, acc3, g, TW, MUL_W, RB)


def row_groups(ids, n_exp: int, rb: int, nch: int = 4):
    """Rows sorted per expert (vLLM's align with block rb) cut into groups of up to nch * rb rows of one expert.
    Returns sid, (group expert, first row, rows), number of groups, grid bound: all on the GPU, no host sync."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    sid, eid, npad = moe_align_block_size(ids, rb, n_exp, None, ignore_invalid_experts=True)
    cnt = torch.bincount(ids.flatten().long(), minlength=n_exp)
    blocks = (cnt + rb - 1) // rb                         # rb-row blocks per expert
    ngrp = (blocks + nch - 1) // nch
    gend = torch.cumsum(ngrp, 0)
    bstart = torch.cumsum(blocks, 0) - blocks             # first block of each expert in sid
    gmax = n_exp + (ids.numel() + nch * rb - 1) // (nch * rb)
    g = torch.arange(gmax, device=ids.device)
    ge = torch.searchsorted(gend, g, right=True).clamp_max(n_exp - 1)
    local = g - (gend[ge] - ngrp[ge])
    gb = ((bstart[ge] + local * nch) * rb).to(torch.int32)
    gn = (torch.minimum(blocks[ge] - local * nch, torch.full_like(local, nch)) * rb).to(torch.int32)
    return sid, ge.to(torch.int32), gb, gn, gend[-1:].to(torch.int32), gmax


def moe_gemm_rg(a, w, s, g, rg, top_k, out, tw=None, RB=32, BN=64, BK=64, num_warps=4, num_stages=2):
    sid, ge, gb, gn, ng, gmax = rg
    n, k = w.shape[1], w.shape[2] * 2
    s8 = s.view(torch.uint8)
    grid = (gmax * triton.cdiv(n, BN),)
    _w4a16_moe_rg[grid](a, w, s8, g, out, sid, ge, gb, gn, ng, tw if tw is not None else g,
                        n, k, out.shape[0],
                        a.stride(0), w.stride(0), w.stride(1), s8.stride(0), s8.stride(1), out.stride(0),
                        TOP_K=top_k, MUL_W=tw is not None, RB=RB, BN=BN, BK=BK,
                        num_warps=num_warps, num_stages=num_stages)
    return out


def moe_rg(x, ids, wts, w13, w2, cfg1: dict, cfg2: dict):
    t, topk = ids.shape
    e = w13[0].shape[0]
    inter = w2[0].shape[2] * 2
    rg1 = row_groups(ids, e, cfg1["RB"])
    h = torch.empty((t * topk, 2 * inter), dtype=torch.bfloat16, device=x.device)
    moe_gemm_rg(x, *w13, rg1, topk, h, **cfg1)
    act = torch.empty((t * topk, inter), dtype=torch.bfloat16, device=x.device)
    torch.ops._C.silu_and_mul(act, h)
    rg2 = rg1 if cfg2["RB"] == cfg1["RB"] else row_groups(ids, e, cfg2["RB"])
    o = torch.empty((t * topk, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    moe_gemm_rg(act, *w2, rg2, 1, o, tw=wts.flatten(), **cfg2)
    return o.view(t, topk, -1).sum(1)
