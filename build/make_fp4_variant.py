#!/usr/bin/env python3
"""Write a checkpoint variant where chosen MXFP8 layer families are re-quantized to W4A16_NVFP4 (the format the
checkpoint already uses for its vision MLP): weight U8 [N, K/2] (two e2m1 codes per byte, low nibble first),
weight_scale F8_E4M3 [N, K/16], weight_scale_2 F32 [] with w ~= e2m1 * weight_scale * weight_scale_2.

The MXFP8 weight is dequantized exactly (e4m3 * 2^(e8m0 - 127)), then quantized to NVFP4 by round-to-nearest with
a per-16 scale chosen among a few candidates per block (min squared error). Layers vLLM fuses (q/k/v + indexer,
gate/up, GDN in_proj qkv/z and a/b) share one weight_scale_2 so a fused GEMM can use a single global scale.
Shards without converted tensors are hard-linked; the others are rewritten. config.json's quantized_layers entries
of the converted layers become {"group_size": 16, "quant_algo": "W4A16_NVFP4"}.

Run inside the vLLM image (needs torch + safetensors): make_fp4_variant.py SRC DST --families gdn,attn,shared
[--device cuda]"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FAMILIES = {
    "gdn": re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"),
    "gdnout": re.compile(r"\.linear_attn\.out_proj$"),
    "attn": re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj)$"),
    "shared": re.compile(r"\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$"),
}
FUSED = [("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"), ("gate_proj", "up_proj"),
         ("in_proj_qkv", "in_proj_z"), ("in_proj_a", "in_proj_b")]
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def group_key(layer):
    """Layers sharing one weight_scale_2: same parent module, members of one fused set."""
    parent, leaf = layer.rsplit(".", 1)
    if leaf == "index_qk_proj":
        parent, leaf = parent.rsplit(".", 1)[0], "indexer.index_qk_proj"
    for i, s in enumerate(FUSED):
        if leaf in s:
            return f"{parent}#{i}"
    return layer


def dequant_mxfp8(w, s):
    w = w.float()
    e = s.to(torch.int32) - 127
    return (w.view(w.shape[0], -1, 32) * torch.pow(2.0, e.float()).unsqueeze(-1)).view(w.shape)


def to_e2m1_codes(x):
    """x already divided by its scale; nearest e2m1 magnitude, sign in bit 3."""
    mag = x.abs().clamp(max=6.0)
    idx = (mag.unsqueeze(-1) - E2M1.to(x.device)).abs().argmin(-1)
    return (idx | ((x < 0).to(idx.dtype) << 3)).to(torch.uint8), E2M1.to(x.device)[idx] * x.sign()


def quant_nvfp4(w, gscale):
    """w [N, K] fp32, gscale = weight_scale_2 -> (packed U8 [N, K/2], scale e4m3 [N, K/16])."""
    N, K = w.shape
    b = w.view(N, K // 16, 16)
    amax = b.abs().amax(-1, keepdim=True)
    best_err, best_s, best_codes = None, None, None
    for f in (1.0, 0.95, 0.9, 0.85, 0.8):                 # candidate block scales, keep the least squared error
        s = (amax * f / 6.0 / gscale).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
        sv = s.float() * gscale
        codes, deq = to_e2m1_codes(b / sv.clamp(min=1e-30))
        err = ((deq * sv - b) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_s, best_codes = err, s, codes
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s.float(), best_s.float()).to(torch.float8_e4m3fn)
            best_codes = torch.where(m, codes, best_codes)
    c = best_codes.view(N, K)
    packed = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    return packed, best_s.view(N, K // 16).contiguous()


def nvfp4_scales(w, gscale):
    """The per-16 block scales quant_nvfp4 picks (min squared error among candidates), as e4m3 [N, K/16]."""
    return quant_nvfp4(w, gscale)[1]


@torch.no_grad()
def gptq_nvfp4(W, H, sc_blk, gscale, damp=0.01, block=128):
    """GPTQ with fixed NVFP4 block scales: W [N, K] fp32, H [K, K], sc_blk e4m3 [N, K/16]. Returns codes U8 [N, K]."""
    N, K = W.shape
    sc = (sc_blk.float() * gscale).repeat_interleave(16, dim=1)          # [N, K] effective scale per weight
    W, H = W.clone(), H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = torch.argsort(torch.diag(H), descending=True)
    W, H, sc = W[:, perm], H[perm][:, perm], sc[:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    C = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    grid = E2M1.to(W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, s = W1[:, i], sc[:, i1 + i]
            x = w / s.clamp(min=1e-30)
            idx = (x.abs().clamp(max=6.0).unsqueeze(-1) - grid).abs().argmin(-1)
            qv = grid[idx] * torch.sign(x)
            C[:, i1 + i] = (idx | ((x < 0).to(idx.dtype) << 3)).to(torch.uint8)
            e = (w - qv * s) / Hi[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return C[:, torch.argsort(perm)]


GPTQ_SRC = {"in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz", "in_proj_a": "in_proj_ba",
            "in_proj_b": "in_proj_ba", "out_proj": "out_proj"}


class Hessians:
    """Lazy {(layer, vllm module leaf): (H from 80 % of the recorded rows, held-out rows)}, 2 entries kept on the GPU
    (tensors of one layer arrive together), so this can run next to a serve."""

    def __init__(self, cal_dir, device):
        import glob
        self.device, self.files, self.cache = device, {}, {}
        for p in glob.glob(os.path.join(cal_dir, "*.pt")):
            m = re.search(r"layers\.(\d+)\.linear_attn\.(in_proj_qkvz|in_proj_ba|out_proj)\.(\d+)\.pt$", p)
            if m:
                self.files.setdefault((int(m.group(1)), m.group(2)), []).append(p)

    def __len__(self):
        return len(self.files)

    def __contains__(self, key):
        return key in self.files

    def __getitem__(self, key):
        if key not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(next(iter(self.cache)))
                torch.cuda.empty_cache()
            X = torch.cat([torch.load(q) for q in sorted(self.files[key])]).to(self.device).float()
            g = torch.Generator(device="cpu").manual_seed(0)
            idx = torch.randperm(X.shape[0], generator=g).to(self.device)
            nf = int(0.8 * X.shape[0])
            self.cache[key] = (X[idx[:nf]].T @ X[idx[:nf]] / nf, X[idx[nf:]])
        return self.cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--families", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gptq", default="", help="calibration dir (recorder files): GPTQ codes for the GDN projections")
    a = ap.parse_args()
    fams = [FAMILIES[f] for f in a.families.split(",")]
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    ql = cfg["quantization_config"]["quantized_layers"]
    conv = sorted(k for k, v in ql.items() if v.get("quant_algo") == "MXFP8" and "language_model" in k
                  and any(f.search(k) for f in fams))
    print(f"{len(conv)} layers to convert ({a.families})", flush=True)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    # global scales per fused group, from the dequantized weights' amax
    groups = {}
    for k in conv:
        groups.setdefault(group_key(k), []).append(k)
    os.makedirs(os.path.join(a.dst, "tensors"), exist_ok=True)
    for f in os.listdir(a.src):
        if f not in ("tensors", "config.json", "model.safetensors.index.json") and not f.startswith("."):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
    by_file = {}
    for k in conv:
        by_file.setdefault(wm[k + ".weight"], []).append(k)
    gamax = {}
    for g, ks in groups.items():
        m = 0.0
        for k in ks:
            with safe_open(os.path.join(a.src, wm[k + ".weight"]), "pt", device=a.device) as fh:
                w = dequant_mxfp8(fh.get_tensor(k + ".weight"), fh.get_tensor(k + ".weight_scale").view(torch.uint8))
            m = max(m, w.abs().max().item())
        gamax[g] = m
    shards = sorted(set(wm.values()))
    errs, gerrs = [], []
    HS = Hessians(a.gptq, a.device) if a.gptq else None
    if HS is not None:
        print(f"{len(HS)} calibrated modules", flush=True)
    for sh in shards:
        src, dst = os.path.join(a.src, sh), os.path.join(a.dst, sh)
        if sh not in by_file:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(os.path.realpath(src), dst)
            continue
        out = {}
        with safe_open(src, "pt", device="cpu") as fh:
            meta = fh.metadata()
            todo = set(by_file[sh])
            for t in fh.keys():
                layer = t.rsplit(".", 1)[0]
                if layer in todo:
                    continue
                out[t] = fh.get_tensor(t)
            for k in sorted(todo):
                w = dequant_mxfp8(fh.get_tensor(k + ".weight").to(a.device),
                                  fh.get_tensor(k + ".weight_scale").view(torch.uint8).to(a.device))
                gs = gamax[group_key(k)] / (6.0 * 448.0)
                pk, sc = quant_nvfp4(w, gs)
                leaf = k.rsplit(".", 1)[1]
                lm = re.search(r"layers\.(\d+)\.", k)
                hk = (int(lm.group(1)), GPTQ_SRC.get(leaf)) if lm else None
                if HS is not None and hk in HS:
                    H, Xh = HS[hk]
                    codes = gptq_nvfp4(w, H, sc, gs)
                    pk_g = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
                    def deq(p):
                        cc = torch.stack([p & 0xF, p >> 4], -1).view(w.shape[0], -1)
                        d = E2M1.to(cc.device)[(cc & 7).long()] * (1 - 2 * ((cc >> 3) & 1).float())
                        return (d.view(w.shape[0], -1, 16) * (sc.float() * gs).unsqueeze(-1)).view(w.shape)
                    ref = Xh @ w.T
                    er = ((Xh @ deq(pk).T - ref).norm() / ref.norm()).item()
                    eg = ((Xh @ deq(pk_g).T - ref).norm() / ref.norm()).item()
                    gerrs.append((k, er, eg))
                    if eg < er:
                        pk = pk_g
                lo, hi = pk & 0xF, pk >> 4
                codes = torch.stack([lo, hi], -1).view(w.shape[0], -1)
                deq = (E2M1.to(codes.device)[(codes & 7).long()] * (1 - 2 * ((codes >> 3) & 1).float()))
                deq = deq.view(w.shape[0], -1, 16) * (sc.float() * gs).unsqueeze(-1)
                rel = ((deq.view(w.shape) - w).norm() / w.norm()).item()
                errs.append((k, rel))
                out[k + ".weight"] = pk.cpu()
                out[k + ".weight_scale"] = sc.cpu()
                out[k + ".weight_scale_2"] = torch.tensor(gs, dtype=torch.float32)
                wm[k + ".weight_scale_2"] = sh
        save_file(out, dst, metadata=meta)
        print(f"{sh}: {len(by_file[sh])} layers converted", flush=True)
    for k in conv:
        ql[k] = {"group_size": 16, "quant_algo": "W4A16_NVFP4"}
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    r = [e for _, e in errs]
    print(f"done: {len(errs)} layers, NVFP4 vs MXFP8 rel.err mean {sum(r) / len(r):.4f} max {max(r):.4f}")
    json.dump(errs, open(os.path.join(a.dst, "fp4_conversion_errors.json"), "w"), indent=1)
    if gerrs:
        import math
        rt = math.sqrt(sum(e * e for _, e, _ in gerrs) / len(gerrs)); gp = math.sqrt(sum(g * g for _, _, g in gerrs) / len(gerrs))
        print(f"GPTQ: {len(gerrs)} tensors, held-out output rel.err RMS RTN {rt:.5f} -> GPTQ {gp:.5f} ({rt / gp:.2f}x)")
        json.dump(gerrs, open(os.path.join(a.dst, "gptq_output_errors.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
