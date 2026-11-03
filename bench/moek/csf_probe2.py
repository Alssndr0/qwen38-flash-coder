import torch, gc
import problem as P
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mu
import mbx_marlin_csf as c
E = 4
w13 = P.real_weights(10)[0]
w2 = P.real_weights(10)[1]
w13 = tuple(t[:E].contiguous() for t in w13); w2 = tuple(t[:E].contiguous() for t in w2)
class L(P.FakeLayer):
    num_experts = E
lay = L()
before = set(dir(lay))
out = mu.prepare_nvfp4_moe_layer_for_marlin(lay, w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], True)
print("new layer attrs", set(dir(lay)) - before)
print("_saved", c._saved, "ratio", c._saved[1] / c._saved[0])
# tensors alive on cuda that are not the outputs or inputs
ids = {id(t) for t in out if t is not None} | {id(t) for t in w13 + w2}
for o in gc.get_objects():
    try:
        if torch.is_tensor(o) and o.is_cuda and id(o) not in ids and o.numel() > 1000:
            print("live", tuple(o.shape), o.dtype, o.numel() * o.element_size())
    except Exception:
        pass
import inspect
for n in ("compress", "pack", "install", "_carrier"):
    f = getattr(c, n)
    try:
        print(n, inspect.signature(f))
    except Exception as ex:
        print(n, "sig?", ex, f.__code__.co_argcount, f.__code__.co_varnames[:f.__code__.co_argcount])
