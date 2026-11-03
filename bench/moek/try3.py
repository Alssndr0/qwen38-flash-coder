"""Ablation of the v1 kernel (gate_up, BM 64 / BN 128 / BK 64): where does the time go?
  DEC=0 current decode (join + fp32 math)   DEC=1 bf16 bit construction (fewer ops)   DEC=2 no decode (wrong values)
  HALF=1 weight layout with byte b of a BK block holding k0+b (low) and k0+b+BK/2 (high): two dots, no interleave"""
import torch
import triton
import triton.language as tl

import problem as P


@triton.jit
def _dec(x, DEC: tl.constexpr):
    if DEC == 0:
        mag = x & 7
        ex = mag >> 1
        m1 = mag & 1
        two_v = tl.where(ex == 0, m1, (2 + m1) << (ex - 1)).to(tl.float32)
        v = tl.where((x & 8) != 0, -0.5 * two_v, 0.5 * two_v).to(tl.bfloat16)
    elif DEC == 1:
        # bf16 bits of value * 2^-6: normal codes put (e, m) under exponent bias 120; codes 0/1 are 0 / 2^-7
        mag = (x & 7).to(tl.int16)
        bits = tl.where(mag < 2, mag * 0x3C00, (mag << 6) + 0x3C00) | ((x & 8).to(tl.int16) << 12)
        v = bits.to(tl.bfloat16, bitcast=True)
    else:
        v = x.to(tl.bfloat16)
    return v


@triton.jit
def _k(A, W, S, G, C, SID, EID, NPAD, N, K, num_valid, stride_am, stride_we, stride_wn, stride_se, stride_sn,
       stride_cm, TOP_K: tl.constexpr, DEC: tl.constexpr, HALF: tl.constexpr,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    mb = pid // num_n
    nb = pid % num_n
    if mb * BM >= tl.load(NPAD):
        return
    e = tl.load(EID + mb).to(tl.int64)
    offs_m = mb * BM + tl.arange(0, BM)
    sid = tl.load(SID + offs_m)
    mvalid = sid < num_valid
    arow = (sid // TOP_K).to(tl.int64)
    offs_n = (nb * BN + tl.arange(0, BN)).to(tl.int64)
    offs_kh = tl.arange(0, BK // 2)
    offs_g = tl.arange(0, BK // 16)
    offs_g2 = tl.arange(0, BK // 32)
    w_ptr = W + e * stride_we + offs_n[:, None] * stride_wn + offs_kh[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if DEC == 1:
        smul = 64.0
    else:
        smul = 1.0
    if HALF:
        a_lo = A + arow[:, None] * stride_am + offs_kh[None, :]
        s_lo = S + e * stride_se + offs_n[:, None] * stride_sn + offs_g2[None, :]
        for _ in range(0, K, BK):
            q = tl.load(w_ptr)
            sl = (tl.load(s_lo).to(tl.float8e4nv, bitcast=True).to(tl.float32) * smul).to(tl.bfloat16)
            sh = (tl.load(s_lo + BK // 32).to(tl.float8e4nv, bitcast=True).to(tl.float32) * smul).to(tl.bfloat16)
            wl = (_dec(q & 0xF, DEC).reshape(BN, BK // 32, 16) * sl[:, :, None]).reshape(BN, BK // 2)
            wh = (_dec(q >> 4, DEC).reshape(BN, BK // 32, 16) * sh[:, :, None]).reshape(BN, BK // 2)
            al = tl.load(a_lo, mask=mvalid[:, None], other=0.0)
            ah = tl.load(a_lo + BK // 2, mask=mvalid[:, None], other=0.0)
            acc = tl.dot(al, tl.trans(wl), acc)
            acc = tl.dot(ah, tl.trans(wh), acc)
            a_lo += BK
            w_ptr += BK // 2
            s_lo += BK // 16
    else:
        offs_k = tl.arange(0, BK)
        a_ptr = A + arow[:, None] * stride_am + offs_k[None, :]
        s_ptr = S + e * stride_se + offs_n[:, None] * stride_sn + offs_g[None, :]
        for _ in range(0, K, BK):
            q = tl.load(w_ptr)
            sc = (tl.load(s_ptr).to(tl.float8e4nv, bitcast=True).to(tl.float32) * smul).to(tl.bfloat16)
            code = tl.join(q & 0xF, q >> 4).reshape(BN, BK)
            w = (_dec(code, DEC).reshape(BN, BK // 16, 16) * sc[:, :, None]).reshape(BN, BK)
            a = tl.load(a_ptr, mask=mvalid[:, None], other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
            a_ptr += BK
            w_ptr += BK // 2
            s_ptr += BK // 16
    if DEC == 1:
        acc = acc * (tl.load(G + e) / 4096.0)    # (value * 2^-6) * (scale * 2^6) needs no fix; 2^-12 from both
    else:
        acc = acc * tl.load(G + e)
    tl.store(C + sid.to(tl.int64)[:, None] * stride_cm + offs_n[None, :], acc.to(tl.bfloat16), mask=mvalid[:, None])


def half_layout(w, bk):
    """[E, N, K/2] (pairs 2i, 2i+1) -> per BK block: byte b = (k0+b) low, (k0+b+BK/2) high."""
    lo, hi = w & 0xF, w >> 4
    v = torch.stack([lo, hi], -1).reshape(w.shape[0], w.shape[1], -1)        # [E, N, K] codes
    v = v.reshape(w.shape[0], w.shape[1], -1, 2, bk // 2)                      # [E, N, K/BK, 2, BK/2]
    return (v[:, :, :, 0] | (v[:, :, :, 1] << 4)).reshape(w.shape[0], w.shape[1], -1).contiguous()


def run(x, w, s, g, sid, eid, npad, top_k, out, DEC, HALF, BM=64, BN=128, BK=64, nw=4, ns=3):
    n, k = w.shape[1], w.shape[2] * 2
    s8 = s.view(torch.uint8)
    grid = (triton.cdiv(sid.numel(), BM) * triton.cdiv(n, BN),)
    _k[grid](x, w, s8, g, out, sid, eid, npad, n, k, out.shape[0], x.stride(0), w.stride(0), w.stride(1),
             s8.stride(0), s8.stride(1), out.stride(0), TOP_K=top_k, DEC=DEC, HALF=HALF, BM=BM, BN=BN, BK=BK,
             num_warps=nw, num_stages=ns)


t = 6912
w13 = P.make_weights(2 * P.INTER, P.HID, 1)
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=0.6)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # noqa: E402
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
import vllm._custom_ops as ops  # noqa: E402

# Marlin's gate_up alone, row tile 48 and 64
mp = P.marlin_prepare(w13, P.make_weights(P.HID, P.INTER, 2))
for bm in (48, 64):
    sid, eid, npad = moe_align_block_size(ids, bm, P.E, None, ignore_invalid_experts=True)
    h = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
    ws = torch.zeros(2 ** 16, dtype=torch.int32, device=P.DEV)
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
    ws = get_marlin_workspace(x.device)
    f = lambda: ops.moe_wna16_marlin_gemm(x, h, mp[0], None, mp[1], None, mp[2], None, ws, sid, eid, npad, wts,
                                          moe_block_size=bm, top_k=P.TOPK, mul_topk_weights=False,
                                          b_q_type=scalar_types.float4_e2m1f, size_m=t, size_n=2 * P.INTER,
                                          size_k=P.HID, use_atomic_add=False, use_fp32_reduce=True,
                                          is_zp_float=False)
    print(f"marlin gate_up bm {bm}: {P.bench(f):.2f} ms")

sid, eid, npad = moe_align_block_size(ids, 64, P.E, None, ignore_invalid_experts=True)
h = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
ref = torch.empty_like(h)
run(x, *w13, sid, eid, npad, P.TOPK, ref, 0, 0)
wh = half_layout(w13[0], 64)
for DEC in (0, 1, 2):
    for HALF in (0, 1):
        w = wh if HALF else w13[0]
        run(x, w, w13[1], w13[2], sid, eid, npad, P.TOPK, h, DEC, HALF)
        ok = (h.float() - ref.float()).norm() / ref.float().norm() if DEC != 2 else float("nan")
        ms = P.bench(lambda: run(x, w, w13[1], w13[2], sid, eid, npad, P.TOPK, h, DEC, HALF))
        print(f"DEC={DEC} HALF={HALF}: {ms:.2f} ms  (rel diff vs DEC0 {ok:.5f})")
