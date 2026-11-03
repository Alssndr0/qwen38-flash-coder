"""What the serve's compressed-scale (csf) Marlin prep stores, vs vLLM's plain Marlin prep."""
import os
import sys
import torch
import problem as P
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mu
import mbx_marlin_csf as c

print("CSF env", os.environ.get("MBX_MARLIN_CSF"))
for k in ("_carrier", "_carriers", "_saved", "LIB"):
    print(k, repr(getattr(c, k))[:300])
for n in ("compress", "install", "pack"):
    f = getattr(c, n)
    print(n, getattr(f, "__doc__", None), getattr(f, "__code__", None) and f.__code__.co_varnames)
E = 8
w13 = P.make_weights(2 * P.INTER, P.HID, 1, e=E)
w2 = P.make_weights(P.HID, P.INTER, 2, e=E)

class L(P.FakeLayer):
    num_experts = E

out = mu.prepare_nvfp4_moe_layer_for_marlin(L(), w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], True)
for name, t in zip(("w13", "s13", "g13", "w2", "s2", "g2"), out):
    print(name, None if t is None else (tuple(t.shape), t.dtype, t.flatten()[:6].tolist() if t.dtype != torch.float8_e4m3fn else t.view(torch.uint8).flatten()[:8].tolist()))
torch.save({k: v.cpu() if v is not None else None for k, v in zip(("w13", "s13", "g13", "w2", "s2", "g2"), out)},
           f"/kit/cache/moek-csf{os.environ.get('MBX_MARLIN_CSF')}.pt")
print("after prep: _saved", c._saved)
for k, v in c._carriers.items():
    print("carrier", k, type(v))
    items = v.items() if isinstance(v, dict) else enumerate(v if isinstance(v, (list, tuple)) else [v])
    for kk, vv in items:
        print("   ", kk, (tuple(vv.shape), vv.dtype, vv.numel() * vv.element_size()) if torch.is_tensor(vv) else repr(vv)[:200])
