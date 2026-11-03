#!/usr/bin/env python3
"""Exactness across boots (for changes that cannot be switched live, e.g. a different expert kernel).

save:    python3 exact_boot.py save NAME [--bg N]   -> results/exact-boot/NAME.json
         per prompt: top-20 logprobs at the last --positions prompt positions (prompt_logprobs: the prefill path),
         and a greedy continuation of --tokens tokens with top-20 logprobs per step (the decode path).
         --bg N runs N background decoders meanwhile (batching noise: the run-to-run floor a user gets anyway).
compare: python3 exact_boot.py compare A B [C D ...]   pairs (A, B), (C, D) ...: next-token KL (mean / p99 / max over
         prompt positions), top-1 agreement, greedy first divergence and the KL per greedy step before it.
"""
import argparse
import json
import math
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc  # noqa: E402
import exact2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "exact-boot")
PROMPTS = ["b1.opencode.local-dash-tests#0", "b1.opencode.local-opt-refactor#3", "b1.opencode.local-recipe-explain#1",
           "b1.pi.swe-django-django-14999#3", "b1.opencode.local-vllm-rejection#5",
           "b1.opencode.swe-django-django-11848#23"]


def save(name, bg, positions, tokens):
    rows = {r["id"]: r for r in conc.load_rows()}
    res = {}
    for pid in PROMPTS:
        ids = exact2.tokenize(rows[pid])

        def run():
            body = {"model": conc.MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                    "cache_salt": uuid.uuid4().hex, "prompt_logprobs": 20}
            pl = exact2.post("/v1/completions", body)["choices"][0]["prompt_logprobs"]
            g = exact2.complete(ids, tokens, uuid.uuid4().hex, logprobs=20)
            return pl, g

        pl, g = exact2.with_bg(bg, rows, run) if bg else run()
        keep = [p for p in pl[-positions:] if p]
        res[pid] = {"n_prompt": len(ids), "prompt": [{k: v["logprob"] for k, v in p.items()} for p in keep],
                    "prompt_ids": ids[-len(keep):], "gen_ids": g["token_ids"],
                    "gen": g["logprobs"]["top_logprobs"]}
        print(f"{name} {pid}: {len(ids)} prompt tokens, {len(keep)} positions, {len(g['token_ids'])} generated",
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(res, f)


def stats(a, b):
    kls, top1, gk, div = [], [], [], []
    for pid in a:
        if pid not in b:
            continue
        pa, pb = a[pid]["prompt"], b[pid]["prompt"]
        for x, y in zip(pa, pb):
            kls.append(exact2.kl(x, y))
            top1.append(max(x, key=x.get) == max(y, key=y.get))
        ga, gb = a[pid]["gen_ids"], b[pid]["gen_ids"]
        d = next((i for i, (u, v) in enumerate(zip(ga, gb)) if u != v), min(len(ga), len(gb)))
        div.append(d)
        for i in range(d):
            gk.append(exact2.kl(a[pid]["gen"][i], b[pid]["gen"][i]))
    kls.sort()
    p99 = kls[int(0.99 * (len(kls) - 1))] if kls else 0.0
    return {"positions": len(kls), "kl_mean": sum(kls) / max(1, len(kls)), "kl_p99": p99, "kl_max": max(kls or [0]),
            "top1_agree": sum(top1) / max(1, len(top1)), "greedy_first_divergence": div,
            "greedy_kl_mean": sum(gk) / max(1, len(gk))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "compare"])
    ap.add_argument("names", nargs="+")
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    if a.cmd == "save":
        save(a.names[0], a.bg, a.positions, a.tokens)
        return
    for x, y in zip(a.names[::2], a.names[1::2]):
        da = json.load(open(os.path.join(OUT, x + ".json")))
        db = json.load(open(os.path.join(OUT, y + ".json")))
        s = stats(da, db)
        print(f"{x} vs {y}: {s['positions']} positions, KL mean {s['kl_mean']:.2e} p99 {s['kl_p99']:.2e} max "
              f"{s['kl_max']:.2e}, top-1 agree {s['top1_agree'] * 100:.2f} %, greedy first divergence "
              f"{s['greedy_first_divergence']} (KL per step before it {s['greedy_kl_mean']:.2e})")


if __name__ == "__main__":
    main()
