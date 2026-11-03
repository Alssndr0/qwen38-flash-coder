#!/usr/bin/env python3
"""Exactness check for scheduling changes that must not change outputs (e.g. layer-sliced prefill).

For each case, with B background streams decoding, the probe request (a real OpenCode prompt) is sent greedy
(temperature 0, logprobs) under each live policy, with a fresh cache_salt per run so its prompt is prefilled from
scratch every time (delta cases: the earlier turn is warmed under the same salt first). Compared per case: the
greedy tokens (first divergence), the chosen-token logprob difference, and the newcomer's tokens per delivery
(drafter acceptance after the prefill: a wrong drafter context shows up there). An A/A pair (the baseline policy
twice) gives the noise floor.

usage: exact.py --policies 'NAME=JSON' 'NAME=JSON' ... [--cases fresh10k,delta3k@40k] [--bg 2] [--tokens 96]
       (the first policy is the baseline and is run twice)"""
import argparse
import json
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "..", "cache", "mbx-smooth.json")


GDN8 = os.path.join(HERE, "..", "cache", "mbx-gdn8.json")


def set_policy(p):
    p = dict(p)
    g = p.pop("gdn8_min", None)  # mbx_gdn8 live threshold (a huge value = 4-bit everywhere)
    if g is not None:
        with open(GDN8 + ".tmp", "w") as f:
            json.dump({"min_tokens": int(g)}, f)
        os.replace(GDN8 + ".tmp", GDN8)
    with open(LIVE + ".tmp", "w") as f:
        json.dump(p, f)
    os.replace(LIVE + ".tmp", LIVE)
    time.sleep(2.5 if g is not None else 1.5)


def probe_body(row, tokens, salt):
    b = conc.body_of(row, tokens)
    b["temperature"] = 0.0
    b["logprobs"] = True
    b["top_logprobs"] = 1
    b["cache_salt"] = salt
    return b


def run_probe(body):
    """stream and keep token ids + logprobs"""
    import http.client
    rec = {"t_send": time.time(), "ids": [], "lps": [], "ev": []}
    c = http.client.HTTPConnection(conc.HOST, conc.PORT, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    for raw in r:
        if not raw.startswith(b"data: ") or raw.startswith(b"data: [DONE]"):
            continue
        d = json.loads(raw[6:])
        for ch in d.get("choices") or []:
            ids = ch.get("token_ids") or []
            if not ids:
                continue
            if "t_first" not in rec:
                rec["t_first"] = time.time()
            rec["ev"].append(len(ids))
            rec["ids"].extend(ids)
            for lp in ((ch.get("logprobs") or {}).get("content") or []):
                rec["lps"].append(lp.get("logprob"))
    return rec


def one(case, row_map, tokens, bg_n, salt):
    spec = conc.CASES[case]
    if spec[0] == "delta":
        w = probe_body(row_map[spec[1]], 1, salt)
        run_probe(w)
    stop = threading.Event()
    bg = []
    for i in range(bg_n):
        rec = {}
        bg.append(rec)
        conc.start(conc.body_of(row_map[conc.BG_IDS[i]], 3000, 3000), rec, stop)
    conc.wait_first(bg)
    time.sleep(2.0)
    p = run_probe(probe_body(row_map[spec[-1]], tokens, salt))
    stop.set()
    time.sleep(1.0)
    return p


def compare(a, b):
    n = min(len(a["ids"]), len(b["ids"]))
    div = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
    m = div if div is not None else n
    d = [abs(x - y) for x, y in zip(a["lps"][:m], b["lps"][:m]) if x is not None and y is not None]
    return {"same_tokens": div is None and len(a["ids"]) == len(b["ids"]), "first_divergence": div,
            "lp_absdiff_mean": sum(d) / len(d) if d else 0.0, "lp_absdiff_max": max(d) if d else 0.0,
            "tpd_common": (tpd_upto(a, m), tpd_upto(b, m))}


def tpd_upto(p, m):
    """tokens per delivery over the first m tokens (the first delivery, the prefill's token, excluded)"""
    n = got = 0
    for e in p["ev"][1:]:
        if got >= m:
            break
        n += 1
        got += e
    return got / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--cases", default="fresh10k,delta3k@40k,delta12k@5k")
    ap.add_argument("--bg", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pols = [(p.split("=", 1)[0], json.loads(p.split("=", 1)[1])) for p in a.policies]
    rows = {r["id"]: r for r in conc.load_rows()}
    runs = [(pols[0][0] + "#A", pols[0][1])] + [(pols[0][0] + "#B", pols[0][1])] + pols[1:]
    res = {}
    for case in a.cases.split(","):
        res[case] = {}
        for name, pol in runs:
            set_policy(pol)
            p = one(case, rows, a.tokens, a.bg, uuid.uuid4().hex)
            res[case][name] = p
            tpd = sum(p["ev"][1:]) / max(len(p["ev"]) - 1, 1)
            print(f"  {case:14s} {name:14s} ttft {p.get('t_first', 0) - p['t_send']:6.2f}s  tokens {len(p['ids'])}"
                  f"  tok/delivery {tpd:.2f}", flush=True)
    set_policy(pols[0][1])
    print("\nvs " + runs[0][0])
    for case, d in res.items():
        base = d[runs[0][0]]
        for name, _ in runs[1:]:
            c = compare(base, d[name])
            print(f"  {case:14s} {name:14s} same={c['same_tokens']!s:5s} first_div={c['first_divergence']!s:5s} "
                  f"|dlogprob| mean {c['lp_absdiff_mean']:.4f} max {c['lp_absdiff_max']:.4f}  "
                  f"tok/delivery on the common text {c['tpd_common'][0]:.2f} -> {c['tpd_common'][1]:.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
