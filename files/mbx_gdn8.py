"""mbx_gdn8: prefill-size GDN projections on the checkpoint's MXFP8 originals, decode stays on the 4-bit GPTQ weights.

v4gdnq serves the GDN projections (in_proj_qkvz, out_proj of the 36 linear-attention layers) as GPTQ NVFP4 through
Marlin W4A16: the right choice for decode (it reads half the bytes of MXFP8), but on a prefill chunk Marlin is
compute-bound at ~75 TFLOP/s, about half of what the MXFP8 W8A8 tensor-core GEMM reaches on GB10. This module loads the
same projections a second time from the AZ checkpoint (the MXFP8 weights NVIDIA shipped, +2.1 GB) and, per call, runs
batches of at least MBX_GDN8_MIN_TOKENS tokens on them; smaller batches (decode, draft verification, small prompts)
keep the 4-bit path unchanged. Measured on GB10 (bench/gdn8_bench.py, MXFP8 incl. the activation quantize): qkvz
1.27x at 256 rows, 1.8x from 512 rows (6,912 rows: 7.7 -> 4.4 ms); out_proj 1.5x at 256 rows, 1.1-1.35x above;
decode sizes (<= 128 rows) are 1.1-3.5x faster on the 4-bit path. Not output-exact for prefill: those tokens see the MXFP8 projections (as on the f5c
serve) instead of the 4-bit ones.

Switches (all off unless MBX_GDN8 is set):
  MBX_GDN8=<dir>             checkpoint holding the MXFP8 projections (its model.safetensors.index.json), e.g.
                             /models/Qwen3.8-Flash-Next-FP4-Local-AZ
  MBX_GDN8_MIN_TOKENS=256    batches from this many tokens use MXFP8 (read at every call: the live file below wins)
  MBX_GDN8_BACKEND=b12x      FlashInfer mm_mxfp8 backend (b12x | cutlass)
  MBX_GDN8_LIVE=<json file>  optional live override {"min_tokens": N} (re-read every 2 s; 0 or absent keeps the env)
Called from base_loader.py (after process_weights_after_loading, target only).
"""
import json
import os
import time

import torch

from vllm.logger import init_logger

log = init_logger("vllm.mbx_gdn8")

_REG: list = []          # key -> _Entry
_STATE = {"min": int(os.environ.get("MBX_GDN8_MIN_TOKENS", "256")), "t": 0.0}
_LIVE = os.environ.get("MBX_GDN8_LIVE", "")
_BACKEND = os.environ.get("MBX_GDN8_BACKEND", "b12x")


class _Entry:
    __slots__ = ("layer", "orig", "w8", "s8", "n", "k", "name", "split")

    def __init__(self, layer, orig, w8, s8, n, k, name, split):
        self.layer, self.orig, self.w8, self.s8, self.n, self.k, self.name = layer, orig, w8, s8, n, k, name
        self.split = split  # above this many rows run in 2048-row pieces (0 = never)


def _min_tokens() -> int:
    if _LIVE:
        now = time.monotonic()
        if now - _STATE["t"] > 2.0:
            _STATE["t"] = now
            try:
                with open(_LIVE) as f:
                    v = int(json.load(f).get("min_tokens") or 0)
                _STATE["live"] = v
            except (OSError, ValueError, AttributeError):
                _STATE["live"] = 0
        if _STATE.get("live"):
            return _STATE["live"]
    return _STATE["min"]


def _mx(e: _Entry, x: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.utils import flashinfer as vfi

    def mm(xp):
        xq, xs = mxfp8_e4m3_quantize(xp, is_sf_swizzled_layout=True)
        return vfi.mm_mxfp8(xq, e.w8.t(), xs, e.s8, out_dtype=x.dtype, backend=_BACKEND)

    m = x.shape[0]
    if not e.split or m <= e.split:
        return mm(x)
    # N 2560 (out_proj): b12x slows down past ~3.5k rows (6,912 rows: 4.8 ms whole, 2.5 ms in 2048-row pieces)
    out = torch.empty((m, e.n), dtype=x.dtype, device=x.device)
    for i in range(0, m, 2048):
        out[i:i + 2048] = mm(x[i:i + 2048])
    return out


def _op(x: torch.Tensor, key: int) -> torch.Tensor:
    e = _REG[key]
    if x.shape[0] >= _min_tokens():
        return _mx(e, x)
    return e.orig.apply(e.layer, x, None)


def _op_fake(x: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], _REG[key].n))


_registered = False


def _register() -> None:
    global _registered
    if _registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="mbx_gdn8_linear", op_func=_op, mutates_args=[], fake_impl=_op_fake)
    _registered = True


class Gdn8Method:
    """Stands in for the layer's quant method: one opaque op that picks the kernel by token count."""

    def __init__(self, orig, key: int) -> None:
        self.orig = orig
        self.key = key

    def __getattr__(self, name):  # anything else (process_weights_after_loading, ...) stays the original's
        return getattr(self.__dict__["orig"], name)

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.mbx_gdn8_linear(x2, self.key)
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out


def _load(src: str, names: list[str]) -> dict:
    from safetensors import safe_open

    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    by_file: dict = {}
    for n in names:
        if n not in wm:
            raise KeyError(f"{n} not in {src}")
        by_file.setdefault(wm[n], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with safe_open(os.path.join(src, fn), framework="pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def target_loaded(model, vllm_config, model_config) -> None:
    src = os.environ.get("MBX_GDN8")
    if not src or "MTP" in type(model).__name__ or model_config is not vllm_config.model_config:
        return
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        log.warning("mbx_gdn8: tensor parallel > 1 is not supported: off")
        return
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    t0 = time.time()
    _register()
    todo = []  # (module name, module, [checkpoint prefixes in output order])
    for name, mod in model.named_modules():
        if not hasattr(mod, "quant_method") or ".linear_attn." not in name or "mtp" in name:
            continue
        idx = name.split(".layers.")[1].split(".")[0]
        base = f"model.language_model.layers.{idx}.linear_attn."
        if name.endswith(".in_proj_qkvz"):
            todo.append((name, mod, [base + "in_proj_qkv", base + "in_proj_z"]))
        elif name.endswith(".out_proj"):
            todo.append((name, mod, [base + "out_proj"]))
    if not todo:
        log.warning("mbx_gdn8: no GDN projections found: off")
        return
    names = [p + s for _, _, ps in todo for p in ps for s in (".weight", ".weight_scale")]
    ts = _load(src, names)
    dev = next(model.parameters()).device
    nbytes = 0
    for name, mod, ps in todo:
        w = torch.cat([ts[p + ".weight"] for p in ps], 0)
        s = torch.cat([ts[p + ".weight_scale"] for p in ps], 0)
        n, k = w.shape
        if n != mod.output_size_per_partition or k != mod.input_size_per_partition:
            raise ValueError(f"mbx_gdn8: {name}: MXFP8 weight {tuple(w.shape)} vs layer "
                             f"({mod.output_size_per_partition}, {mod.input_size_per_partition})")
        w8 = w.to(dev).contiguous()
        s8 = swizzle_mxfp8_scale(s[:, : k // 32].contiguous().to(dev), M=n, K=k).contiguous()
        nbytes += w8.numel() + s8.numel()
        key = len(_REG)
        _REG.append(_Entry(mod, mod.quant_method, w8, s8, n, k, name, 3456 if n <= 4096 else 0))
        mod.quant_method = Gdn8Method(mod.quant_method, key)
    del ts
    # check: the two paths on the same activation agree to quantization noise (a wrong row order would not)
    with torch.no_grad():
        for e in (_REG[0], _REG[1]):
            g = torch.Generator(device=dev).manual_seed(0)
            x = torch.randn(256, e.k, device=dev, dtype=torch.bfloat16, generator=g)
            a = e.orig.apply(e.layer, x, None).float()
            b = _mx(e, x).float()
            rel = ((a - b).norm() / a.norm()).item()
            log.info("mbx_gdn8: %s: MXFP8 vs 4-bit relative difference %.4f", e.name, rel)
            if not rel < 0.25:
                raise RuntimeError(f"mbx_gdn8: {e.name}: MXFP8 and 4-bit outputs differ by {rel:.3f}: wrong layout?")
    log.info("mbx_gdn8: %d GDN projections with MXFP8 copies from %s (%.2f GB) in %.1f s; batches >= %d tokens use "
             "them (backend %s)", len(_REG), src, nbytes / 1e9, time.time() - t0, _STATE["min"], _BACKEND)
