#!/usr/bin/env python3
"""GDN projection GEMMs, Marlin W4A16 NVFP4 (what decode uses) vs MXFP8 W8A8 (the checkpoint's originals), per M.
Runs inside the serve container (random weights: kernel time does not depend on values):
  docker exec -i qwen38-flash-next python3 - < bench/gdn8_bench.py
Shapes: in_proj_qkvz (N 16384, K 2560), out_proj (N 2560, K 6144). MXFP8 time includes the activation quantize."""
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize, swizzle_mxfp8_scale
from vllm.utils import flashinfer as vfi

dev = "cuda"
MS = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 3456, 6912]


class L(torch.nn.Module):
    pass


def marlin_layer(N, K):
    l = L()
    l.weight = torch.nn.Parameter(torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=dev), requires_grad=False)
    l.weight_scale = torch.nn.Parameter((torch.rand(N, K // 16, device=dev) * 2 + 0.5).to(torch.float8_e4m3fn),
                                        requires_grad=False)
    l.weight_global_scale = torch.nn.Parameter(torch.tensor(0.01, device=dev), requires_grad=False)
    l.output_size_per_partition, l.input_size_per_partition, l.params_dtype = N, K, torch.bfloat16
    prepare_fp4_layer_for_marlin(l)
    return l


def mx_weights(N, K):
    w = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 124, (N, K // 32), dtype=torch.uint8, device=dev)
    return w, swizzle_mxfp8_scale(s, M=N, K=K)


def timeit(fn, it=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it * 1000.0  # us


for name, N, K in (("qkvz", 16384, 2560), ("out", 2560, 6144)):
    ml = marlin_layer(N, K)
    w8, s8 = mx_weights(N, K)
    print(f"{name}: N {N} K {K}   M | marlin us | mxfp8 b12x us (quant us) | mxfp8 cutlass us | best")
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        t_m = timeit(lambda: apply_fp4_marlin_linear(x, ml.weight, ml.weight_scale, ml.weight_global_scale, None, N, K))
        t_q = timeit(lambda: mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True))
        res = {}
        for be in ("b12x", "cutlass"):
            def f(be=be):
                xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
                return vfi.mm_mxfp8(xq, w8.t(), xs, s8, out_dtype=torch.bfloat16, backend=be)
            try:
                res[be] = timeit(f)
            except Exception as e:  # noqa: BLE001
                res[be] = float("nan")
                print("   ", be, "failed:", str(e)[:120])
        best = min(res.values())
        print(f"  {M:5d} | {t_m:8.1f} | {res['b12x']:8.1f} ({t_q:6.1f}) | {res['cutlass']:8.1f} | "
              f"{'mxfp8' if best < t_m else 'marlin'} x{t_m / best:.2f}", flush=True)
