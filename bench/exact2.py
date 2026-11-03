#!/usr/bin/env python3
"""Exactness of a prefill path, teacher-forced: the same token sequence scored under each live policy.

Per case: the OpenCode prompt P is tokenized (/tokenize, with its tools), a greedy continuation T is generated once
(baseline policy), then for each cut k the sequence P + T[:k] is sent as a fresh prompt (new cache_salt, so it is
prefilled from scratch; delta cases warm the earlier turn under the same salt first) with max_tokens 1 and the top-20
logprobs of the next token. B background streams decode meanwhile, so slice-type policies actually slice. Compared to
the first baseline run: KL over the top-20 (baseline's), top-1 agreement. A second baseline run is the noise floor.

usage: exact2.py --policies 'NAME=JSON' ... [--cases fresh10k,delta3k@40k,delta12k@5k] [--cuts 40,80,120,160] [--bg 2]
"""
import argparse
import http.client
import json
import math
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "..", "cache", "mbx-smooth.json")


def post(path, body):
    c = http.client.HTTPConnection(conc.HOST, conc.PORT, timeout=3600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise RuntimeError(f"{path}: HTTP {r.status} {data[:300]!r}")
    return json.loads(data)


GDN8 = os.path.join(HERE, "..", "cache", "mbx-gdn8.json")
MOE = os.path.join(HERE, "..", "cache", "mbx-moe.json")


def set_policy(p):
    p = dict(p)
    g = p.pop("gdn8_min", None)  # mbx_gdn8 live threshold (a huge value = 4-bit everywhere)
    if g is not None:
        with open(GDN8 + ".tmp", "w") as f:
            json.dump({"min_tokens": int(g)}, f)
        os.replace(GDN8 + ".tmp", GDN8)
    bm = p.pop("moe_bm", None)  # mbx_moe live Marlin MoE row tile (0 = vLLM's choice)
    if bm is not None:
        with open(MOE + ".tmp", "w") as f:
            json.dump({"bm": int(bm), "min_tokens": 256}, f)
        os.replace(MOE + ".tmp", MOE)
    with open(LIVE + ".tmp", "w") as f:
        json.dump(p, f)
    os.replace(LIVE + ".tmp", LIVE)
    time.sleep(2.5 if g is not None or bm is not None else 1.5)


def tokenize(row):
    b = row["body"]
    req = {"model": conc.MODEL, "messages": b["messages"], "add_generation_prompt": True}
    if b.get("tools"):
        req["tools"] = b["tools"]
    if b.get("chat_template_kwargs"):
        req["chat_template_kwargs"] = b["chat_template_kwargs"]
    if b.get("reasoning_effort"):
        req["chat_template_kwargs"] = dict(req.get("chat_template_kwargs") or {}, reasoning_effort=b["reasoning_effort"])
    return post("/tokenize", req)["tokens"]


def complete(ids, max_tokens, salt, logprobs=None):
    body = {"model": conc.MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "cache_salt": salt, "return_token_ids": True}
    if logprobs:
        body["logprobs"] = logprobs
    return post("/v1/completions", body)["choices"][0]


def with_bg(n, rows, fn):
    stop = threading.Event()
    bg = []
    for i in range(n):
        rec = {}
        bg.append(rec)
        conc.start(conc.body_of(rows[conc.BG_IDS[i]], 4000, 4000), rec, stop)
    conc.wait_first(bg)
    time.sleep(1.5)
    try:
        return fn()
    finally:
        stop.set()
        time.sleep(0.5)


def top(choice):
    lp = choice["logprobs"]["top_logprobs"][0]
    return {k: v for k, v in lp.items()}


def kl(a, b, floor=-20.0):
    """KL(a||b) over a's top tokens, a renormalized; b's missing tokens at floor."""
    za = math.log(sum(math.exp(v) for v in a.values()))
    s = 0.0
    for t, la in a.items():
        pa = math.exp(la - za)
        lb = b.get(t, floor)
        s += pa * ((la - za) - lb)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--cases", default="fresh10k,delta3k@40k,delta12k@5k")
    ap.add_argument("--cuts", default="40,80,120,160")
    ap.add_argument("--bg", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--warm-bg", action="store_true", help="warm the earlier turn with the background running")
    a = ap.parse_args()
    pols = [(p.split("=", 1)[0], json.loads(p.split("=", 1)[1])) for p in a.policies]
    runs = [(pols[0][0] + "#A", pols[0][1]), (pols[0][0] + "#B", pols[0][1])] + pols[1:]
    rows = {r["id"]: r for r in conc.load_rows()}
    cuts = [int(x) for x in a.cuts.split(",")]
    res = {}
    set_policy(pols[0][1])
    for case in a.cases.split(","):
        spec = conc.CASES[case]
        p_ids = tokenize(rows[spec[-1]])
        w_ids = tokenize(rows[spec[1]]) if spec[0] == "delta" else None
        t_ids = complete(p_ids, max(cuts), uuid.uuid4().hex)["token_ids"]
        res[case] = {}
        for name, pol in runs:
            set_policy(pol)
            out = []
            for k in cuts:
                salt = uuid.uuid4().hex
                if w_ids is not None:
                    if a.warm_bg:  # the earlier turn under the same policy and load, so it is sliced too
                        with_bg(a.bg, rows, lambda: complete(w_ids, 1, salt))
                    else:
                        complete(w_ids, 1, salt)
                t0 = time.time()
                ch = with_bg(a.bg, rows, lambda: complete(p_ids + t_ids[:k], 1, salt, logprobs=20))
                out.append({"k": k, "top": top(ch), "ttft": time.time() - t0})
            res[case][name] = out
            print(f"  {case:14s} {name:10s} " + " ".join(f"{o['ttft']:.1f}s" for o in out), flush=True)
    set_policy(pols[0][1])
    print(f"\nKL vs {runs[0][0]} (top-20, nats) per cut; top-1 agreement")
    for case, d in res.items():
        base = d[runs[0][0]]
        for name, _ in runs[1:]:
            ks = [kl(b["top"], o["top"]) for b, o in zip(base, d[name])]
            agree = sum(max(b["top"], key=b["top"].get) == max(o["top"], key=o["top"].get) for b, o in zip(base, d[name]))
            print(f"  {case:14s} {name:10s} KL " + " ".join(f"{x:.4f}" for x in ks) + f"   top1 {agree}/{len(ks)}")
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
