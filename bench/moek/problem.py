"""One routed-expert layer of the model as a standalone problem: random NVFP4 weights in the checkpoint layout, a routing
with the skew of a real MoE, the bf16 reference, and the serve's own Marlin path to beat.

Shapes (Qwen3.8-Flash-Next): 512 experts, 5 per token, hidden 2560, expert intermediate 640 (gate+up = 1280 rows).
Checkpoint layout per expert: weight [N, K/2] uint8 (two e2m1 per byte, element 2i in the low nibble), weight_scale
[N, K/16] float8_e4m3 (one per 16 inputs), weight_scale_2 fp32 (one per expert).
"""
import math
import os

import torch

E, TOPK, HID, INTER = 512, 5, 2560, 640
DEV = "cuda"

# e2m1 code -> value (sign in bit 3)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def make_weights(n: int, k: int, seed: int, e: int = E):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, generator=g)
    # realistic scale magnitudes: weight ~ N(0, 0.02) -> block max ~0.05 -> scale = max / 6 in e4m3 after a global
    # scale that maps the tensor's largest block scale to e4m3's 448
    s = (torch.rand((e, n, k // 16), generator=g) * 0.9 + 0.1) * 300.0
    s = s.to(torch.float8_e4m3fn)
    gs = (torch.rand((e,), generator=g) * 0.5 + 0.75) * (0.05 / 6 / 300.0)
    return w.to(DEV), s.to(DEV), gs.float().to(DEV)


def dequant(w, s, gs, i: int) -> torch.Tensor:
    """Expert i to bf16 [N, K] (fp32 math, exact: e2m1 * e4m3 fits bf16)."""
    lut = E2M1.to(w.device)
    q = w[i]
    lo, hi = lut[(q & 0xF).long()], lut[(q >> 4).long()]
    v = torch.stack([lo, hi], -1).reshape(q.shape[0], -1)  # [N, K]
    sc = s[i].float().repeat_interleave(16, dim=1)
    return (v * sc * gs[i]).to(torch.bfloat16)


def routing(t: int, seed: int = 0, sigma: float = 1.0):
    """Top-5 of 512 per token with lognormal expert popularity (sigma 1: the busiest experts get ~5-8x the mean)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pop = torch.randn(E, generator=g) * sigma
    gum = -torch.log(-torch.log(torch.rand((t, E), generator=g).clamp_min(1e-20)))
    ids = torch.topk(pop[None, :] + gum, TOPK, dim=1).indices.to(torch.int32)
    wts = torch.softmax(torch.randn((t, TOPK), generator=g), dim=1).float()
    return ids.to(DEV), wts.to(DEV)


def reference(x, ids, wts, w13, w2, rows=None):
    """bf16 MoE output for the given token rows (default all), unfused, for correctness."""
    t = x.shape[0]
    rows = range(t) if rows is None else rows
    out = torch.zeros((len(rows), HID), dtype=torch.float32, device=x.device)
    cache: dict = {}
    for j, r in enumerate(rows):
        for kk in range(TOPK):
            e = int(ids[r, kk])
            if e not in cache:
                cache[e] = (dequant(*w13, e), dequant(*w2, e))
            a, b = cache[e]
            h = (x[r:r + 1].float() @ a.float().T)
            h = torch.nn.functional.silu(h[:, :INTER]) * h[:, INTER:]
            h = h.to(torch.bfloat16).float()
            out[j] += float(wts[r, kk]) * (h @ b.float().T)[0]
    return out


class FakeLayer:
    num_experts = E
    hidden_size = HID
    intermediate_size_per_partition = INTER
    params_dtype = torch.bfloat16


def marlin_prepare(w13, w2):
    """The serve's preparation (mbx compressed scales when MBX_MARLIN_CSF=1)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin

    out = prepare_nvfp4_moe_layer_for_marlin(FakeLayer(), w13[0], w13[1], w13[2], w2[0], w2[1], w2[2], True)
    return out  # w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2


def marlin_moe(x, ids, wts, mp, bm: int = 0):
    """The serve's Marlin MoE (fused_marlin_moe's steps), with vLLM's row tile or a forced one."""
    import vllm.model_executor.layers.fused_moe.experts.marlin_moe as mm
    from vllm.scalar_type import scalar_types

    w1, s1, g1, w2, s2, g2 = mp
    t = x.shape[0]
    if not bm:
        for bm in [8, 16, 32, 48, 64]:
            if t * TOPK / E / bm < 0.9:
                break
    sid, eid, npad = mm.moe_align_block_size(ids, bm, E, None, ignore_invalid_experts=True)
    out = mm._fused_marlin_moe(hidden_states=x, w1=w1, w2=w2, bias1=None, bias2=None, w1_scale=s1, w2_scale=s2,
                               topk_weights=wts, num_topk=TOPK, quant_type=scalar_types.float4_e2m1f,
                               apply_router_weight_on_input=False, expert_map=None, block_size_m=bm,
                               sorted_token_ids=sid, expert_ids=eid, num_tokens_post_padded=npad,
                               global_scale1=g1, global_scale2=g2).view(-1, TOPK, HID)
    return out.sum(1)


def bench(fn, iters: int = 10, warm: int = 3) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def floor_ms(t: int) -> dict:
    """Weight-read and tensor-core floors for one layer (230 GB/s, 106 TFLOP/s bf16)."""
    wbytes = E * (2 * INTER * HID + HID * INTER) * (0.5 + 1 / 16)
    flops = 2 * t * TOPK * (2 * INTER * HID + HID * INTER)
    return {"weights_ms": wbytes / 230e9 * 1e3, "mma_ms": flops / 106e12 * 1e3}


if __name__ == "__main__":
    torch.manual_seed(0)
    t = int(os.environ.get("T", "6912"))
    w13 = make_weights(2 * INTER, HID, 1)
    w2 = make_weights(HID, INTER, 2)
    x = (torch.randn((t, HID), device=DEV) * 0.5).to(torch.bfloat16)
    ids, wts = routing(t)
    cnt = torch.bincount(ids.flatten().long(), minlength=E)
    print(f"T={t}: rows per expert mean {cnt.float().mean():.1f} max {int(cnt.max())} min {int(cnt.min())}; "
          f"floors {floor_ms(t)}")
    mp = marlin_prepare(w13, w2)
    rows = list(range(0, t, max(1, t // 24)))[:24]
    ref = reference(x, ids, wts, w13, w2, rows)
    for bm in (0, 48):
        out = marlin_moe(x, ids, wts, mp, bm)
        err = (out[rows].float() - ref).norm() / ref.norm()
        ms = bench(lambda: marlin_moe(x, ids, wts, mp, bm))
        print(f"marlin bm={bm or 'vllm'}: {ms:.2f} ms per layer (whole MoE), rel err vs reference {err:.4f}")


def real_weights(layer: int = 10):
    """One layer's real routed experts from the checkpoint (CSF scales decoded by the image's loader), cached as
    (w13, w2) in make_weights' format. gate and up share one global scale per expert (checked)."""
    path = f"/kit/cache/moek-layer{layer}.pt"
    if os.path.exists(path):
        d = torch.load(path)
        return tuple(t.to(DEV) for t in d["w13"]), tuple(t.to(DEV) for t in d["w2"])
    import json

    import mbx_csf
    from safetensors import safe_open

    root = "/kit/models/Qwen3.8-Flash-Next-FP4-Local-AZ-w4gdnq/"
    ix = json.load(open(root + "model.safetensors.index.json"))["weight_map"]
    pre = f"model.language_model.layers.{layer}.mlp.experts."
    files: dict = {}

    def get(name):
        fn = ix[name]
        if fn not in files:
            files[fn] = safe_open(root + fn, "pt")
        return files[fn].get_tensor(name)

    def proj(e, p):
        b = f"{pre}{e}.{p}."
        s = mbx_csf.decode_fixed(get(b + "weight_scale.nvfp4_csf_fixed"), get(b + "weight_scale.nvfp4_csf_exceptions"))
        return get(b + "weight"), s, get(b + "weight_scale_2").float().reshape(())

    w13, s13, g13, w2, s2, g2 = [], [], [], [], [], []
    for e in range(E):
        (wg, sg, gg), (wu, su, gu), (wd, sd, gd) = proj(e, "gate_proj"), proj(e, "up_proj"), proj(e, "down_proj")
        assert float(gg) == float(gu), (e, float(gg), float(gu))
        w13.append(torch.cat([wg, wu])); s13.append(torch.cat([sg, su])); g13.append(gg)
        w2.append(wd); s2.append(sd); g2.append(gd)
    out = {"w13": (torch.stack(w13), torch.stack(s13), torch.stack(g13)), "w2": (torch.stack(w2), torch.stack(s2), torch.stack(g2))}
    torch.save(out, path)
    return tuple(t.to(DEV) for t in out["w13"]), tuple(t.to(DEV) for t in out["w2"])


def uniform_routing(rows_per_expert: int):
    """Every expert gets exactly rows_per_expert rows; each token 5 distinct experts. T = rows * E / 5."""
    t = rows_per_expert * E // TOPK
    j = torch.arange(t * TOPK, device=DEV)
    ids = ((j * 1) % E).view(t, TOPK).to(torch.int32)
    wts = torch.full((t, TOPK), 1.0 / TOPK, device=DEV)
    return ids, wts


def marlin_plain(layer: int = 10):
    """vLLM's plain-scale Marlin preparation of a real layer (w13, s13, g13, w2, s2, g2), cached on disk so that a
    csf process (MBX_MARLIN_CSF=1) can also feed the kernel; produce it once with MBX_MARLIN_CSF=0."""
    path = f"/kit/cache/moek-layer{layer}-marlin-plain.pt"
    if os.path.exists(path):
        return tuple(t.to(DEV) for t in torch.load(path))
    assert os.environ.get("MBX_MARLIN_CSF") == "0", "run once with CSF=0 to build " + path
    mp = marlin_prepare(*real_weights(layer))
    torch.save([t.cpu() for t in mp], path)
    return mp
