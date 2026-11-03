#!/usr/bin/env python3
"""Exactness of a prefill path when the request itself carries no logprobs (sliced prefill does not slice requests
that ask for logprobs). Two checks, per case:
  seq:   the probe prompt P is generated greedy (K tokens, no logprobs) under each policy; token sequences compared
         (first divergence). Run with --bg 0 and a policy with "force": 1 for a deterministic comparison.
  next:  the earlier turn A is prefilled under the policy (max_tokens 1, no logprobs: so it is sliced), then the later
         turn B, which hits A's prefix cache, is scored normally with top-20 logprobs: KL vs the baseline policy.
usage: exact3.py --policies 'NAME=JSON' ... [--cases ...] [--bg 0] [--tokens 24]"""
import argparse, json, os, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc, exact2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policies", nargs="+", required=True)
ap.add_argument("--cases", default="fresh10k,delta12k@5k,delta3k@40k")
ap.add_argument("--bg", type=int, default=0)
ap.add_argument("--tokens", type=int, default=24)
a = ap.parse_args()
pols = [(p.split("=", 1)[0], json.loads(p.split("=", 1)[1])) for p in a.policies]
runs = [(pols[0][0] + "#A", pols[0][1]), (pols[0][0] + "#B", pols[0][1])] + pols[1:]
rows = {r["id"]: r for r in conc.load_rows()}
off = pols[0][1]
for case in a.cases.split(","):
    spec = conc.CASES[case]
    p_ids = exact2.tokenize(rows[spec[-1]])
    a_ids = exact2.tokenize(rows[spec[1]]) if spec[0] == "delta" else None
    seqs, nxt = {}, {}
    for name, pol in runs:
        exact2.set_policy(pol)
        salt = uuid.uuid4().hex
        if a_ids is not None:
            exact2.with_bg(a.bg, rows, lambda: exact2.complete(a_ids, 1, salt))
        ch = exact2.with_bg(a.bg, rows, lambda: exact2.complete(p_ids, a.tokens, salt))
        seqs[name] = ch["token_ids"]
        if a_ids is not None:   # the later turn scored normally on the (sliced) earlier turn's cache
            exact2.set_policy(off)
            nxt[name] = exact2.top(exact2.complete(p_ids, 1, salt + "x", logprobs=20)) if False else None
    base = seqs[runs[0][0]]
    for name, _ in runs[1:]:
        s = seqs[name]
        div = next((i for i in range(min(len(s), len(base))) if s[i] != base[i]), None)
        print(f"  {case:14s} {name:12s} seq: same={div is None} first_div={div}  ({len(s)} tokens)", flush=True)
exact2.set_policy(off)
