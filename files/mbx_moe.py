"""mbx_moe: the routed experts' Marlin row tile on prefill-size batches. Off unless MBX_MOE_BM or MBX_MOE_LIVE is set.

vLLM picks Marlin's MoE row tile (moe_block_size 8/16/32/48/64) from the average number of rows per expert; at a
6,912-token chunk (5 of 512 experts per token, ~68 rows per expert) that is 64, and every expert's rows are padded up
to a multiple of it. A tile of 48 pads less: prefill 1.6-2.8 % faster on GB10 (bench/moe_sweep.sh, fresh 10k-66k,
reproduced; 16 and 64 = vLLM's choice, 32 halfway).

  MBX_MOE_BM=48              batches of >= MBX_MOE_MIN_TOKENS tokens use this row tile (16 | 32 | 48 | 64)
  MBX_MOE_MIN_TOKENS=256     smaller batches (decode, draft verification, small prompts) keep vLLM's choice
  MBX_MOE_LIVE=<json file>   optional live override {"bm": N, "min_tokens": M}, re-read every 2 s ("bm": 0 = vLLM's
                             choice); a missing file keeps the env values

Outputs: the tile only changes how the work is split (padding rows are dropped), not what is summed; results stay at
the run-to-run level (Marlin's split-K reduction order can differ). Installed from base_loader.py (worker process).
"""
import json
import os
import time


def _env_bm() -> int:
    bm = int(os.environ.get("MBX_MOE_BM", "0") or 0)
    return bm if bm in (16, 32, 48, 64) else 0


_LIVE = os.environ.get("MBX_MOE_LIVE", "")
_ENV = (_env_bm(), int(os.environ.get("MBX_MOE_MIN_TOKENS", "256") or 256))
_S: dict = {"t": 0.0, "cfg": _ENV, "pend": None}


def _cfg() -> tuple[int, int]:
    if not _LIVE:
        return _ENV
    now = time.monotonic()
    if now - _S["t"] > 2.0:
        _S["t"] = now
        try:
            with open(_LIVE) as f:
                d = json.load(f)
            bm = int(d.get("bm") or 0)
            _S["cfg"] = (bm if bm in (16, 32, 48, 64) else 0, int(d.get("min_tokens") or _ENV[1]))
        except (OSError, ValueError, AttributeError, TypeError):
            _S["cfg"] = _ENV
    return _S["cfg"]


def install() -> None:
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm

    if getattr(mm, "_mbx_moe", False):
        return
    orig_align = mm.moe_align_block_size
    orig_fused = mm._fused_marlin_moe

    # fused_marlin_moe() calls moe_align_block_size(topk_ids, block_size_m, ...) and then
    # _fused_marlin_moe(..., block_size_m=...): both must see the same tile, so the choice made at the first call
    # is handed to the second.
    def align(topk_ids, block_size, *a, **k):
        bm, mn = _cfg()
        if bm and topk_ids.shape[0] >= mn:
            block_size = bm
        _S["pend"] = block_size
        return orig_align(topk_ids, block_size, *a, **k)

    def fused(*a, **k):
        b = _S["pend"]
        _S["pend"] = None
        if b is not None and "block_size_m" in k:
            k["block_size_m"] = b
        return orig_fused(*a, **k)

    mm.moe_align_block_size = align
    mm._fused_marlin_moe = fused
    mm._mbx_moe = True
    from vllm.logger import init_logger

    init_logger("vllm.mbx_moe").info("mbx_moe: Marlin MoE row tile %s from %d tokens%s", _ENV[0] or "vLLM's",
                                     _ENV[1], f" (live: {_LIVE})" if _LIVE else "")
