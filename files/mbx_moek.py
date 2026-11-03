"""mbx_moek: the routed experts (NVFP4 weights, bf16 activations) on the kit's own CUDA kernel, files/mbx_moek.cu,
instead of Marlin. Off unless MBX_MOEK=1. The experts' scales come from vLLM's plain Marlin preparation (the kernel
packs them itself); the dense NVFP4 layers keep the image's compressed scales (MBX_MARLIN_CSF).

Why: the image's compressed-scale Marlin (csf) decodes scales once per 48-row block, which makes prefill ~20 % slower
than plain Marlin, and plain Marlin needs 3.2 GB more for the scales. This kernel reads its own compressed scales
(lossless, 0.53 of the plain bytes: 3.93 GB for the model, a little under csf) and decodes them once per weight tile,
so one layer at a 6,912-token chunk takes 9.6 ms instead of 12.2 (bench/moek, layer 10) and decode matches csf.

At load (prepare_nvfp4_moe_layer_for_marlin, wrapped):
  - vLLM's own plain-scale preparation (the module's code without the image's csf patch, _stock_moe_prep);
  - scales -> pack_scales: per 8 positions x 8 k-groups one base + 4-bit offsets, the 0.24 % of scales outside
    their window in per-(expert, item, run) exception lists; tile_scales puts an item's run pair together;
  - weights (Marlin's int32 [E, K/16, 2N]) -> tile_weights, in place: one 8 KB block per (expert, 256 columns,
    64-k stage), same bytes;
  - vLLM keeps the tiled weight in place of the Marlin one and a 1-element placeholder for the scales.
In the forward (fused_marlin_moe, wrapped): weights registered here run moe_align (96-row blocks), the gate_up kernel
(silu_and_mul fused), the down kernel (router weight applied) and vLLM's own top-k sum; anything else stays on Marlin.
Batches of <= MBX_MOEK_SPLIT_T tokens (default 16) split gate_up along k (5 units) so all SMs stream one expert at a
time; their fp32 partials are added by a small second kernel in a fixed order (deterministic).

Outputs: the same arithmetic as Marlin (dequant bit tricks, bf16 scale multiply, fp32 accumulation, global scale and
router weight on the fp32 sum, vLLM's silu_and_mul rounding); sums run in a different order, so results differ from
Marlin at the bf16 rounding level (error vs a bf16 reference equal or lower, bench/moek/correct_sweep.py; greedy
decode across boots within the baseline's own run-to-run spread, bench/exact_boot.py).
Installed from base_loader.py before the weights are processed (worker process).
"""
import inspect
import os

import torch

BM, KG, RUN = 96, 8, 16 + 8 * 64
SPLIT_T = int(os.environ.get("MBX_MOEK_SPLIT_T", "16"))
NS_GATE, NS_DOWN = 5, 1
_BUILD = os.environ.get("MBX_MOEK_BUILD", "/cache/moek-build")
_REG: dict = {}  # tiled w13 data_ptr -> layer state
_BUF: dict = {}
_S = {"mod": None, "layers": 0, "bytes_plain": 0, "bytes_packed": 0}


def _log():
    from vllm.logger import init_logger

    return init_logger("vllm.mbx_moek")


def mod():
    if _S["mod"] is None:
        from torch.utils.cpp_extension import load

        os.makedirs(_BUILD, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbx_moek.cu")
        _S["mod"] = load(name="mbx_moek_ext", sources=[src], build_directory=_BUILD,
                         extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo"],
                         verbose=False)
    return _S["mod"]


# ----------------------------------------------------------------------------------------------- weight preparation
def pack_scales(s, gate_up, chunk=64):
    """pack_scales_part over expert chunks (small temporaries), joined; exception lists re-based."""
    e = s.shape[0]
    if e <= chunk:
        return pack_scales_part(s, gate_up)
    parts = [pack_scales_part(s[e0:min(e, e0 + chunk)], gate_up) for e0 in range(0, e, chunk)]
    packed = torch.cat([p[0] for p in parts])
    offs, ents, base = [], [], 0
    for p in parts:
        n_exc = int(p[1][-1])
        offs.append(p[1][:-1] + base)
        ents.append(p[2][:n_exc])
        base += n_exc
    exc_off = torch.cat(offs + [torch.tensor([base], dtype=torch.int32, device=s.device)])
    ent = torch.cat(ents) if base else parts[0][2][:1]
    return packed, exc_off, ent


def pack_scales_part(s, gate_up):
    """Marlin plain scales [E, K/16, N] -> (packed [E, K/16/KG, N/128, RUN] uint8, exc_off int32, exc_ent int32).

    A block = 8 consecutive positions of a scale row (one kernel lane's 8 scales) x one run of KG k-groups (64
    scales) shares one base: the start of the 16-value window that holds most of them. Per (expert, run, 128
    positions): 16 bases, then KG rows of 64 bytes of 4-bit offsets (byte i of each 8-position chunk: position i low
    nibble, i + 4 high). Scales outside their window are exceptions: offset 0 and an int32 entry (position | k-group
    in the run << 16 | value << 24) in a per-(expert, item, run) list (item = the kernel's 256-column tile:
    gate_up pairs gate run r with up run N/256 + r); exc_off = prefix sums over (expert, item, run).
    """
    b = s.view(torch.uint8)
    e, kr, n = b.shape
    nb = kr // KG
    v = b.view(e, nb, KG, n // 8, 8).permute(0, 1, 3, 2, 4).reshape(e, nb, n // 8, KG * 8).to(torch.int16)
    srt = v.sort(-1).values
    hi = torch.searchsorted(srt, srt + 15, right=True)  # for each sorted start, the end of its 16-value window
    cover = hi - torch.arange(KG * 8, device=s.device)
    base = torch.gather(srt, -1, cover.argmax(-1, keepdim=True))  # [e, nb, n/8, 1]
    off = v - base
    exc = (off < 0) | (off > 15)
    off = torch.where(exc, 0, off).to(torch.uint8).view(e, nb, n // 8, KG, 8).permute(0, 1, 3, 2, 4)  # [e,nb,KG,n/8,8]
    nib = (off[..., :4] | (off[..., 4:] << 4)).reshape(e, nb, KG, n // 128, 64)
    packed = torch.cat([base.view(e, nb, n // 128, 16).to(torch.uint8),
                        nib.permute(0, 1, 3, 2, 4).reshape(e, nb, n // 128, KG * 64)], -1).contiguous()
    idx = exc.nonzero()  # (e, b, chunk, i): i = kg * 8 + position in the chunk
    nt = n // 256
    run = idx[:, 2] // 16
    item = (run % nt) if gate_up else run // 2
    key = (idx[:, 0] * nt + item) * nb + idx[:, 1]
    order = torch.argsort(key, stable=True)
    idx, key = idx[order], key[order]
    cnt = torch.bincount(key, minlength=e * nt * nb)
    exc_off = torch.zeros(e * nt * nb + 1, dtype=torch.int32, device=s.device)
    exc_off[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    ent = torch.zeros(max(idx.shape[0], 1), dtype=torch.int32, device=s.device)
    if idx.shape[0]:
        kg, p8 = idx[:, 3] // 8, idx[:, 3] % 8
        col = idx[:, 2] * 8 + p8
        val = v[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].to(torch.int64)
        word = col | (kg << 16) | (val << 24)  # as uint32; stored in int32 (two's complement)
        ent[:idx.shape[0]] = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return packed, exc_off, ent


def unpack_scales(packed, exc_off, ent, kr, n):
    """Inverse of pack_scales_part on the untiled layout (tests): -> [E, K/16, N] uint8."""
    e, nb = packed.shape[:2]
    base = packed[..., :16].reshape(e, nb, n // 8).to(torch.int16)
    nib = packed[..., 16:].reshape(e, nb, n // 128, KG, 64).permute(0, 1, 3, 2, 4).reshape(e, nb, KG, n // 8, 4)
    off = torch.cat([nib & 0xF, nib >> 4], -1).to(torch.int16)  # [e, nb, KG, n/8, 8]
    v = (off + base[:, :, None, :, None]).reshape(e, nb, KG, n)
    m = int(exc_off[-1])
    if m:
        en = ent[:m].to(torch.int64) & 0xFFFFFFFF
        col, kg, val = en & 0xFFFF, (en >> 16) & 0xFF, (en >> 24) & 0xFF
        key = torch.searchsorted(exc_off[1:].to(torch.int64), torch.arange(m, device=ent.device), right=True)
        nt = n // 256
        v[key // (nt * nb), key % nb, kg, col] = val.to(torch.int16)
    return v.to(torch.uint8).reshape(e, kr, n)


def _item_tiles(n, gate_up, dev):
    nt = n // 256
    if gate_up:  # 128 gate columns + the matching 128 up columns
        t = torch.arange(nt, device=dev)
        return torch.stack([2 * t, 2 * t + 1, n // 128 + 2 * t, n // 128 + 2 * t + 1], 1)
    return torch.arange(n // 64, device=dev).view(nt, 4)


def tile_weights(w, gate_up, chunk=8):
    """Marlin int32 [E, K/16, 2N] -> the kernel's order, IN PLACE (same shape): per (expert, item, stage) 8 KB,
    inside it k16 row r: [4 tiles x 512 B] in the item's tile order. Chunks of experts keep the temporaries small
    (no second full-size copy: less memory fragmentation at load)."""
    e, kr, n2 = w.shape
    n = n2 // 2
    nt = n // 256
    idx = _item_tiles(n, gate_up, w.device).flatten()
    for e0 in range(0, e, chunk):
        e1 = min(e, e0 + chunk)
        g = w[e0:e1].view(e1 - e0, kr, n // 64, 128)[:, :, idx, :].view(e1 - e0, kr // 4, 4, nt, 4, 128)
        w[e0:e1] = g.permute(0, 3, 1, 2, 4, 5).reshape(e1 - e0, kr, n2)
    return w


def tile_scales(packed, n, gate_up):
    """packed [E, nb, N/128, RUN] -> per (expert, item, run) the item's two 128-column runs: [E, N/256, nb, 2 RUN]."""
    e, nb = packed.shape[:2]
    nt = n // 256
    dev = packed.device
    if gate_up:
        ridx = torch.stack([torch.arange(nt, device=dev), n // 256 + torch.arange(nt, device=dev)], 1)
    else:
        ridx = torch.arange(n // 128, device=dev).view(nt, 2)
    return packed[:, :, ridx.flatten(), :].view(e, nb, nt, 2 * RUN).permute(0, 2, 1, 3).contiguous()


def _buffers(dev):
    if "ws" not in _BUF:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        _BUF["ws"] = torch.empty(2 * sms * BM * 256, dtype=torch.float32, device=dev)  # stream-k (unused by default)
        _BUF["cnt"] = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
        # split partials, sized once for the largest split call: units x tokens x top-k x N (here 5 x 12 x 5 x 1280)
        _BUF["part"] = torch.empty(max(NS_GATE, NS_DOWN) * SPLIT_T * 8 * 2560, dtype=torch.float32, device=dev)
        _BUF["none"] = torch.empty(0, dtype=torch.uint8, device=dev)
    return _BUF


def _prep_wrap(orig):
    sig = inspect.signature(orig)

    def prep(*args, **kwargs):
        out = orig(*args, **kwargs)
        w13, s13, g13, w2, s2, g2 = out
        try:
            ba = sig.bind(*args, **kwargs)
            ok = bool(ba.arguments.get("is_act_and_mul", True))
        except TypeError:
            ok = True
        e = w13.shape[0]
        k1, n1 = w13.shape[1] * 16, w13.shape[2] // 2
        k2, n2 = w2.shape[1] * 16, w2.shape[2] // 2
        shapes_ok = (ok and s13 is not None and s2 is not None and s13.dim() == 3 and s13.shape == (e, k1 // 16, n1)
                     and s2.dim() == 3 and s2.shape == (e, k2 // 16, n2) and s13.element_size() == 1
                     and k1 % 128 == 0 and k2 % 128 == 0 and n1 % 256 == 0 and n2 % 256 == 0 and (n1 // 2) % 128 == 0
                     and w13.dtype == torch.int32 and w2.dtype == torch.int32)
        if not shapes_ok:
            if not _S.get("warned"):
                _S["warned"] = True
                _log().warning("mbx_moek: a MoE layer's Marlin preparation is not the plain NVFP4 one (scales %s): "
                               "it stays on Marlin", None if s13 is None else tuple(s13.shape))
            return out
        mod()
        _buffers(w13.device)
        sdt = s13.dtype
        _S["bytes_plain"] += s13.numel() + s2.numel()
        p13, o13, x13 = pack_scales(s13, True)
        p2, o2, x2 = pack_scales(s2, False)
        p13, p2 = tile_scales(p13, n1, True), tile_scales(p2, n2, False)
        del s13, s2
        _S["bytes_packed"] += p13.numel() + p2.numel() + o13.numel() * 4 + o2.numel() * 4 + x13.numel() + x2.numel()
        w13t = tile_weights(w13, True)
        w2t = tile_weights(w2, False)
        del w13, w2, out
        placeholder = torch.zeros(1, dtype=sdt, device=w13t.device)
        _REG[w13t.data_ptr()] = {"w2": w2t.data_ptr(), "sp13": p13, "off13": o13, "ent13": x13, "sp2": p2,
                                 "off2": o2, "ent2": x2}
        _S["layers"] += 1
        return w13t, placeholder, g13, w2t, placeholder.clone(), g2

    prep._mbx_moek = True
    return prep


# --------------------------------------------------------------------------------------------------------- forward
def _fused_wrap(orig):
    sig = inspect.signature(orig)

    def fused(*args, **kwargs):
        ba = sig.bind(*args, **kwargs)
        a = ba.arguments
        w1 = a["w1"]
        r = _REG.get(w1.data_ptr())
        if r is None:
            return orig(*args, **kwargs)
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

        x, w2 = a["hidden_states"], a["w2"]
        topk_ids, topk_w = a["topk_ids"], a["topk_weights"]
        act_kind = a.get("activation")
        if (a.get("bias1") is not None or a.get("bias2") is not None or a.get("expert_map") is not None
                or a.get("apply_router_weight_on_input") or w2.data_ptr() != r["w2"]
                or (act_kind is not None and "SILU" not in str(act_kind).upper())
                or a.get("global_num_experts", -1) not in (-1, w1.shape[0])
                or (a.get("input_dtype") is not None and a["input_dtype"].itemsize == 1)):
            raise RuntimeError("mbx_moek: unsupported fused_marlin_moe call for kernel weights: "
                               f"activation {act_kind}, expert_map {a.get('expert_map') is not None}, "
                               f"bias {a.get('bias1') is not None}, input_dtype {a.get('input_dtype')}")
        m, k = x.shape
        output = a.get("output")
        if m == 0:  # empty batch (vLLM dummy runs)
            return output if output is not None else torch.empty_like(x)
        e = w1.shape[0]
        topk = topk_ids.shape[1]
        n1 = w1.shape[2] // 2  # gate+up rows
        inter = n1 // 2
        ids = m * topk
        buf = _BUF
        sid, eid, npad = moe_align_block_size(topk_ids, BM, e, None, ignore_invalid_experts=True)
        # own buffers: vLLM's modular-kernel output may alias its intermediate caches
        act = torch.empty((ids, inter), dtype=x.dtype, device=x.device)
        dout = torch.empty((ids, k), dtype=x.dtype, device=x.device)
        split = m <= SPLIT_T
        ids32 = topk_ids.reshape(-1) if topk_ids.dtype == torch.int32 else topk_ids.reshape(-1).to(torch.int32)
        tw = topk_w.reshape(-1)
        md = mod()
        none = buf["none"]
        g1, g2 = a["global_scale1"], a["global_scale2"]
        md.moe_gemm(x, w1, none, g1, sid, eid, npad, g1, act, topk, True, r["sp13"], r["off13"], r["ent13"],
                    buf["ws"], buf["cnt"], 2 if split else 0, NS_GATE if split else 1, buf["part"], 0, True)
        if split:
            md.moe_split_epilogue(buf["part"], NS_GATE, ids32, g1, g1, act, True, n1)
        md.moe_gemm(act, w2, none, g2, sid, eid, npad, tw, dout, 1, False, r["sp2"], r["off2"], r["ent2"],
                    buf["ws"], buf["cnt"], 2 if split else 0, NS_DOWN if split else 1, buf["part"], 0, True)
        if split:  # mode 2 always leaves fp32 partials, also with a single unit per item
            md.moe_split_epilogue(buf["part"], NS_DOWN, ids32, g2, tw, dout, False, k)
        if output is None:
            output = torch.empty_like(x)
        moe_sum = a.get("moe_sum")
        if moe_sum is None:
            return torch.sum(dout.view(-1, topk, k), dim=1, out=output)
        return moe_sum(dout.view(-1, topk, k), output, topk_ids, None)

    fused._mbx_moek = True
    return fused


def _stock_moe_prep(mu):
    """vLLM's own prepare_nvfp4_moe_layer_for_marlin (plain Marlin scales). The image's compressed-scale patch
    (mbx_marlin_csf, appended to the module as "# mbx patch 38") replaces it with a version whose scales live inside
    the closed .so; the kernel needs the plain ones. The dense NVFP4 layers keep the compressed scales."""
    src = open(mu.__file__).read()
    cut = src.find("# mbx patch")
    if cut < 0:
        return mu.prepare_nvfp4_moe_layer_for_marlin
    ns = {"__name__": mu.__name__ + "._stock", "__file__": mu.__file__, "__builtins__": __builtins__}
    exec(compile(src[:cut], mu.__file__, "exec"), ns)
    return ns["prepare_nvfp4_moe_layer_for_marlin"]


def install() -> None:
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as mu

    if getattr(mm.fused_marlin_moe, "_mbx_moek", False):
        return
    prep = _prep_wrap(_stock_moe_prep(mu))
    mu.prepare_nvfp4_moe_layer_for_marlin = prep
    try:
        import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as onv

        onv.prepare_nvfp4_moe_layer_for_marlin = prep
    except ImportError:
        pass
    mm.fused_marlin_moe = _fused_wrap(mm.fused_marlin_moe)
    mod()  # build (first boot) or load the extension now, not in the middle of the weight processing
    _log().info("mbx_moek: routed experts on the kit kernel (split-k up to %d tokens)", SPLIT_T)


def report() -> None:
    if _S["layers"]:
        _log().info("mbx_moek: %d MoE layers converted; expert scales %.2f GB plain -> %.2f GB packed",
                    _S["layers"], _S["bytes_plain"] / 1e9, _S["bytes_packed"] / 1e9)
