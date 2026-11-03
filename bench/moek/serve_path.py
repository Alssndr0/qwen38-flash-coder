"""mbx_moek's forward (the serve's code path, files/mbx_moek.py) vs Marlin on one real layer, many sizes: the
harness test of the integration itself."""
import sys
import torch
sys.path.insert(0, "/kit/files")
import problem as P
import mbx_moek as M
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mu
import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
from vllm.scalar_type import scalar_types

M._BUILD = "/kit/cache/moek-build-serve"
w13, w2 = P.real_weights(10)
ref_mp = P.marlin_plain(10)
fused = M._fused_wrap(mm.fused_marlin_moe)
prep = M._prep_wrap(M._stock_moe_prep(mu))
out = prep(layer=P.FakeLayer(), w13=w13[0], w13_scale=w13[1], w13_scale_2=w13[2], w2=w2[0], w2_scale=w2[1],
           w2_scale_2=w2[2], is_act_and_mul=True)
for t in (1, 5, 8, 12, 13, 31, 128, 700, 6912):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, seed=t, sigma=0.6)
    y = fused(hidden_states=x, w1=out[0], w2=out[3], bias1=None, bias2=None, w1_scale=out[1], w2_scale=out[4],
              topk_weights=wts, topk_ids=ids, quant_type_id=scalar_types.float4_e2m1f.id, global_scale1=out[2],
              global_scale2=out[5])
    ym = P.marlin_moe(x, ids, wts, ref_mp, 0)
    d = ((y.float() - ym.float()).norm() / ym.float().norm()).item()
    print(f"T={t}: serve path vs Marlin rel diff {d:.2e} {'OK' if d < 1e-2 else 'WRONG'}", flush=True)
