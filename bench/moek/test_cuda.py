"""CUDA kernel vs Marlin (plain scales) on one real layer: correctness, then times."""
import os
import sys

import torch

import kcuda
import problem as P
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace
from vllm.scalar_type import scalar_types

assert os.environ.get("MBX_MARLIN_CSF") == "0", "plain Marlin scales needed (CSF=0)"
t = int(os.environ.get("T", "6912"))
w13, w2 = P.real_weights(int(os.environ.get("LAYER", "10")))
mp = P.marlin_prepare(w13, w2)
x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
ids, wts = P.routing(t, sigma=float(os.environ.get("SIGMA", "0.6")))
cnt = torch.bincount(ids.flatten().long(), minlength=P.E)
print(f"T={t}: rows per expert mean {cnt.float().mean():.1f} max {int(cnt.max())}, experts > 128 rows: "
      f"{int((cnt > 128).sum())}")
kcuda.mod()
ws = get_marlin_workspace(x.device)
sid48, eid48, np48 = kcuda.align(ids, P.E, 48)
hm = torch.empty((t * P.TOPK, 2 * P.INTER), dtype=torch.bfloat16, device=P.DEV)
mgu = lambda: ops.moe_wna16_marlin_gemm(x, hm, mp[0], None, mp[1], None, mp[2], None, ws, sid48, eid48, np48, wts,
                                        moe_block_size=48, top_k=P.TOPK, mul_topk_weights=False,
                                        b_q_type=scalar_types.float4_e2m1f, size_m=t, size_n=2 * P.INTER,
                                        size_k=P.HID, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
mgu()
sid, eid, npad = kcuda.align(ids, P.E)
act = kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK)
actm = torch.empty_like(act)
torch.ops._C.silu_and_mul(actm, hm)
torch.cuda.synchronize()
d = (act.float() - actm.float()).norm() / actm.float().norm()
exact = (act == actm).float().mean()
print(f"gate_up+silu vs Marlin+silu_and_mul: rel diff {d:.2e}, bit-identical {exact * 100:.2f} %")
if d > 1e-2:
    sys.exit("gate_up wrong")
out = kcuda.moe(x, ids, wts, mp)
ref_m = P.marlin_moe(x, ids, wts, mp, 48)
rows = list(range(0, t, max(1, t // 24)))[:24]
ref = P.reference(x, ids, wts, w13, w2, rows)
e1 = (out[rows].float() - ref).norm() / ref.norm()
e2 = (ref_m[rows].float() - ref).norm() / ref.norm()
e3 = (out.float() - ref_m.float()).norm() / ref_m.float().norm()
print(f"layer vs reference: kernel {e1:.4f}, Marlin {e2:.4f}; kernel vs Marlin {e3:.2e}")
if e1 > 0.02:
    sys.exit("layer wrong")
print(f"gate_up+silu: Marlin bm48 {P.bench(lambda: (mgu(), torch.ops._C.silu_and_mul(actm, hm))):.2f} ms, kernel "
      f"{P.bench(lambda: kcuda.gate_up(x, mp, sid, eid, npad, P.TOPK, act)):.2f} ms "
      f"(weight floor {P.floor_ms(t)['weights_ms'] * 2 / 3:.2f})")
print(f"down: kernel {P.bench(lambda: kcuda.down(act, mp, sid, eid, npad, wts)):.2f} ms "
      f"(weight floor {P.floor_ms(t)['weights_ms'] / 3:.2f})")
print(f"layer: Marlin bm48 {P.bench(lambda: P.marlin_moe(x, ids, wts, mp, 48)):.2f} ms, "
      f"kernel {P.bench(lambda: kcuda.moe(x, ids, wts, mp)):.2f} ms")
