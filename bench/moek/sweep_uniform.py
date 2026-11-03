"""Kernel vs Marlin (plain) gate_up and down on uniform routings: streaming efficiency vs rows per expert."""
import os
import torch
import kcuda
import problem as P
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
from vllm.scalar_type import scalar_types

w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
ws = get_marlin_workspace(torch.device("cuda"))
kcuda.mod()
wb13 = P.E * 2 * P.INTER * P.HID * (0.5 + 1 / 16)
wb2 = wb13 / 2
for rows in [int(r) for r in os.environ.get("ROWS", "16,48,96,128").split(",")]:
    ids, wts = P.uniform_routing(rows)
    t = ids.shape[0]
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    bm = 48 if rows >= 48 else 16
    s48, e48, n48 = kcuda.align(ids, P.E, bm)
    hm = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
    mg = lambda: ops.moe_wna16_marlin_gemm(x, hm, mp[0], None, mp[1], None, mp[2], None, ws, s48, e48, n48, wts,
                                           moe_block_size=bm, top_k=P.TOPK, mul_topk_weights=False,
                                           b_q_type=scalar_types.float4_e2m1f, size_m=t, size_n=2 * P.INTER,
                                           size_k=P.HID, use_atomic_add=False, use_fp32_reduce=True,
                                           is_zp_float=False)
    om = torch.empty((t * P.TOPK, P.HID), dtype=torch.bfloat16, device=P.DEV)
    md = lambda: ops.moe_wna16_marlin_gemm(act, om, mp[3], None, mp[4], None, mp[5], None, ws, s48, e48, n48, wts,
                                           moe_block_size=bm, top_k=1, mul_topk_weights=True,
                                           b_q_type=scalar_types.float4_e2m1f, size_m=t * P.TOPK, size_n=P.HID,
                                           size_k=P.INTER, use_atomic_add=False, use_fp32_reduce=True,
                                           is_zp_float=False)
    sid, eid, npad = kcuda.align(ids, P.E)
    h = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
    act = torch.randn((t * P.TOPK, P.INTER), device=P.DEV).to(torch.bfloat16) * 0.1
    o = kcuda.down(act, mp, sid, eid, npad, wts)
    a, b, c, d = (P.bench(mg), P.bench(lambda: kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK, h)), P.bench(md),
                  P.bench(lambda: kcuda.down(act, mp, sid, eid, npad, wts, o)))
    print(f"{rows:4d} rows/expert: gate_up Marlin {a:5.2f} ({wb13 / a / 1e6:3.0f} GB/s) kernel {b:5.2f} "
          f"({wb13 / b / 1e6:3.0f}) | down Marlin {c:5.2f} ({wb2 / c / 1e6:3.0f}) kernel {d:5.2f} ({wb2 / d / 1e6:3.0f})")
