#!/usr/bin/env python3
"""Sampling calibration on the real model: is the serve sampling from the distribution it reports?
Sample coding answers at temperature 1, top-k 20, top-p 1 with top-20 logprobs. Sampling is then from the renormalized
top 20, so for an unbiased sampler the token chosen has rank r as often as those probabilities say. Reports observed vs
expected frequency per rank group with a z score (sum of Bernoulli variances). usage: calib.py NAME [--n 60] [--c 4]
-> results/calib/NAME.json"""
import argparse, concurrent.futures as cf, json, math, os, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("LAB_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
M = os.environ.get("LAB_MODEL", "Qwen/Qwen3.8-Flash-Next")
TOPICS = ["an LRU cache with TTL", "a token bucket rate limiter", "merging overlapping intervals", "a thread-safe bounded queue",
          "retry with exponential backoff and jitter", "parsing a cron expression", "topological sort with cycle detection",
          "a consistent hash ring", "streaming JSON lines with error recovery", "a B-tree insert", "a Raft leader election timer",
          "an idempotent database migration runner", "a circuit breaker", "diffing two directory trees", "a trie-based router"]
LANGS = ["Python", "Go", "TypeScript", "Rust"]
def prompts(n):
    return [f"Implement {TOPICS[i % len(TOPICS)]} in {LANGS[i % len(LANGS)]}. Then review your own code for bugs and edge "
            f"cases, and explain the trade-offs. Variant {i}." for i in range(n)]
GROUPS = (("top1", 0, 1), ("top2", 1, 2), ("top3-5", 2, 5), ("top6-20", 5, 20))
def one(p, a):
    b = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": a.tokens, "temperature": 1.0, "top_p": 1.0,
         "top_k": 20, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}}
    o = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
    S = {g[0]: [0, 0.0, 0.0] for g in GROUPS}; skipped = 0; n = 0
    for t in (o["choices"][0].get("logprobs") or {}).get("content") or []:
        top = sorted(t["top_logprobs"], key=lambda x: -x["logprob"])
        ids = [x["token"] for x in top]
        if t["token"] not in ids:
            skipped += 1; continue
        ps = [math.exp(x["logprob"]) for x in top]; z = sum(ps); ps = [x / z for x in ps]; r = ids.index(t["token"]); n += 1
        for g, lo, hi in GROUPS:
            pg = sum(ps[lo:hi]); s = S[g]; s[0] += lo <= r < hi; s[1] += pg; s[2] += pg * (1 - pg)
    return S, n, skipped
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--c", type=int, default=4); ap.add_argument("--tokens", type=int, default=600); a = ap.parse_args()
    T = {g[0]: [0, 0.0, 0.0] for g in GROUPS}; N = sk = 0
    with cf.ThreadPoolExecutor(a.c) as ex:
        for S, n, s in ex.map(lambda p: one(p, a), prompts(a.n)):
            N += n; sk += s
            for k in T:
                for i in range(3): T[k][i] += S[k][i]
    out = {"tokens": N, "not_in_top20": sk, "groups": {}}
    print(f"{a.name}: {N} tokens, not in top 20: {sk}")
    for k, (obs, exp, var) in T.items():
        z = (obs - exp) / math.sqrt(var) if var else 0
        out["groups"][k] = {"observed": obs / max(N, 1), "expected": exp / max(N, 1), "z": z}
        print(f"  {k:8s} observed {100 * obs / max(N, 1):6.2f} %  expected {100 * exp / max(N, 1):6.2f} %  z {z:+.1f}")
    os.makedirs(os.path.join(HERE, "results", "calib"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "calib", a.name + ".json"), "w"), indent=1)
if __name__ == "__main__":
    main()
