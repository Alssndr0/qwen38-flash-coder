"""Where Marlin loses time: real layer weights; csf vs plain at decode and prefill sizes; uniform 48 vs 96 rows."""
import os
import torch
import problem as P
import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
from vllm.scalar_type import scalar_types

csf = os.environ.get("MBX_MARLIN_CSF")
w13, w2 = P.real_weights(int(os.environ.get("LAYER", "10")))
mp = P.marlin_prepare(w13, w2)
ws = get_marlin_workspace(torch.device("cuda"))


def gate_up(x, ids, wts, bm):
    t = x.shape[0]
    sid, eid, npad = moe_align_block_size(ids, bm, P.E, None, ignore_invalid_experts=True)
    h = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
    return lambda: ops.moe_wna16_marlin_gemm(x, h, mp[0], None, mp[1], None, mp[2], None, ws, sid, eid, npad, wts,
                                             moe_block_size=bm, top_k=P.TOPK, mul_topk_weights=False,
                                             b_q_type=scalar_types.float4_e2m1f, size_m=t, size_n=2 * P.INTER,
                                             size_k=P.HID, use_atomic_add=False, use_fp32_reduce=True,
                                             is_zp_float=False)


wb = P.E * 2 * P.INTER * P.HID * (0.5 + 1 / 16)
for t in (8, 24, 64, 6912):
    ids, wts = P.routing(t, sigma=0.6)
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ms = P.bench(lambda: P.marlin_moe(x, ids, wts, mp, 48 if t >= 256 else 0), iters=20)
    touched = torch.unique(ids).numel()
    print(f"csf={csf} T={t}: whole MoE {ms:.3f} ms (experts touched {touched}: weight floor "
          f"{touched / P.E * P.floor_ms(t)['weights_ms']:.3f} ms)")
for rows in (16, 32, 48, 96, 144):
    ids, wts = P.uniform_routing(rows)
    t = ids.shape[0]
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    for bm in (16, 48, 64):
        ms = P.bench(gate_up(x, ids, wts, bm))
        print(f"csf={csf} uniform {rows} rows/expert (T={t}) bm={bm}: gate_up {ms:.2f} ms = "
              f"{wb / ms / 1e6:.0f} GB/s of weights once, {2 * t * 5 * 1280 * 2560 / ms / 1e9:.1f} TFLOP/s")
