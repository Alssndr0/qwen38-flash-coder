"""One kernel call per variant for Nsight Compute (WHAT=kernel|marlin), real layer, T=6912 sigma 0.6."""
import os
import torch
import kcuda
import problem as P
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
from vllm.scalar_type import scalar_types

t = int(os.environ.get("T", "6912"))
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=0.6)
if os.environ.get("UNIFORM"):
    ids, wts = P.uniform_routing(int(os.environ["UNIFORM"]))
    t = ids.shape[0]
    x = x[:t].contiguous() if x.shape[0] >= t else (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
what = os.environ.get("WHAT", "kernel")
if what == "kernel":
    kcuda.mod()
    sid, eid, npad = kcuda.align(ids, P.E)
    h = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
    for _ in range(3):
        kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK, h)
else:
    ws = get_marlin_workspace(x.device)
    sid, eid, npad = kcuda.align(ids, P.E, 48)
    hm = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
    for _ in range(4):
        ops.moe_wna16_marlin_gemm(x, hm, mp[0], None, mp[1], None, mp[2], None, ws, sid, eid, npad, wts,
                                  moe_block_size=48, top_k=P.TOPK, mul_topk_weights=False,
                                  b_q_type=scalar_types.float4_e2m1f, size_m=t, size_n=2 * P.INTER, size_k=P.HID,
                                  use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
torch.cuda.synchronize()
