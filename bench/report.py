#!/usr/bin/env python3
"""Summaries of conc.py results (and the serve's step trace) per trial; several trials side by side.

usage: report.py TRIAL [TRIAL ...] [--json] [--steps]
  Prints, per scenario, the smoothness and speed measures. --steps adds the step-time table from the step trace
  (cache/steptrace, the steps that fall inside each scenario's run). --json writes results/<trial>/summary.json too.

Measures:
  stall:    newcomer TTFT (s); keep% = decode speed of the running streams while the newcomer prefills, as % of their
            speed in the 3 s before it arrived; freeze = longest gap between two token deliveries on a running stream
            in that window (ms).
  agents:   speed = decode tokens/s per stream (all decode time pooled), total = completion tokens / wall time,
            TTFT p50/p90, gaps (ms) between deliveries within a reply: p50/p99/max, gaps over 0.5 s and 1 s per
            1000 tokens, stall% = share of decode time spent in the part of gaps beyond 250 ms.
  capacity: per-stream tokens/s at c = 1..C, total.
  prefill:  prompt tokens / TTFT for a fresh prompt alone.
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "..", "cache", "steptrace")


def load(trial, scen):
    p = os.path.join(HERE, "results", trial, f"{scen}.jsonl")
    if not os.path.exists(p):
        return None, None
    recs = [json.loads(l) for l in open(p)]
    meta = json.load(open(os.path.join(HERE, "results", trial, f"{scen}.meta.json")))
    return recs, meta


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def tokens_between(ev, t0, t1):
    return sum(n for t, n, _ in ev if t0 <= t < t1)


def max_gap(ev, t0, t1):
    """Longest wait between deliveries in [t0, t1], counting from the last delivery before t0."""
    ts = [t for t, _, _ in ev]
    before = [t for t in ts if t < t0]
    inside = [t for t in ts if t0 <= t <= t1]
    seq = ([before[-1]] if before else []) + inside
    after = [t for t in ts if t > t1]
    if after:
        seq.append(after[0])
    if len(seq) < 2:
        return t1 - t0
    return max(b - a for a, b in zip(seq, seq[1:]))


def stall_summary(recs):
    by = {}
    for r in recs:
        by.setdefault((r["case"], r["rep"]), []).append(r)
    rows = {}
    for (case, rep), rs in by.items():
        inj = [r for r in rs if r["role"] == "inject"][0]
        bg = [r for r in rs if r["role"] == "bg" and r["ev"]]
        if not inj.get("t_first"):
            continue
        w0, w1 = inj["t_send"], inj["t_first"]
        keep, gaps = [], []
        for b in bg:
            base = tokens_between(b["ev"], w0 - 3.0, w0) / 3.0
            win = tokens_between(b["ev"], w0, w1) / max(w1 - w0, 1e-6)
            if base > 0:
                keep.append(100.0 * win / base)
            gaps.append(1000.0 * max_gap(b["ev"], w0, w1))
        u = inj.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        d = rows.setdefault(case, {"ttft": [], "keep": [], "freeze": [], "new_tokens": []})
        d["ttft"].append(w1 - w0)
        d["keep"].append(st.mean(keep) if keep else float("nan"))
        d["freeze"].append(max(gaps) if gaps else float("nan"))
        d["new_tokens"].append(u.get("prompt_tokens", 0) - cached)
    return {k: {m: st.mean(v) for m, v in d.items()} for k, d in rows.items()}


def agents_summary(recs, meta):
    dec_tok = dec_time = 0.0
    n_ev = 0
    ttft, gaps, out_tok = [], [], 0
    big1 = big05 = 0
    stall_s = 0.0
    errs = 0
    for r in recs:
        if r.get("err"):
            errs += 1
        ev = r.get("ev") or []
        u = r.get("usage") or {}
        out_tok += u.get("completion_tokens", sum(n for _, n, _ in ev))
        if not ev:
            continue
        ttft.append(ev[0][0] - r["t_send"])
        if len(ev) > 1:
            dec_tok += sum(n for _, n, _ in ev[1:])
            n_ev += len(ev) - 1
            dec_time += ev[-1][0] - ev[0][0]
            for (a, _, _), (b, _, _) in zip(ev, ev[1:]):
                g = b - a
                gaps.append(1000.0 * g)
                big05 += g > 0.5
                big1 += g > 1.0
                stall_s += max(0.0, g - 0.25)
    wall = meta["t1"] - meta["t0"]
    return {
        "requests": len(recs), "errors": errs, "wall_s": wall, "out_tokens": out_tok,
        "total_tps": out_tok / wall, "stream_tps": dec_tok / max(dec_time, 1e-6),
        "ttft_p50": pct(ttft, 0.5), "ttft_p90": pct(ttft, 0.9), "ttft_max": max(ttft) if ttft else float("nan"),
        "gap_p50": pct(gaps, 0.5), "gap_p99": pct(gaps, 0.99), "gap_max": max(gaps) if gaps else float("nan"),
        "gaps05_per_ktok": 1000.0 * big05 / max(dec_tok, 1), "gaps1_per_ktok": 1000.0 * big1 / max(dec_tok, 1),
        "stall_pct": 100.0 * stall_s / max(dec_time, 1e-6),
        "tok_per_delivery": dec_tok / max(n_ev, 1),
    }


def capacity_summary(recs, meta):
    out = {}
    secs = meta["secs"]
    for c in sorted({r["c"] for r in recs}):
        rs = [r for r in recs if r["c"] == c and r["ev"]]
        t0 = max(r["t_first"] for r in rs) + 1.0
        rates = [tokens_between(r["ev"], t0, t0 + secs) / secs for r in rs]
        gaps = [1000.0 * max_gap(r["ev"], t0, t0 + secs) for r in rs]
        out[c] = {"stream_tps": st.mean(rates), "total_tps": sum(rates), "max_gap_ms": max(gaps)}
    return out


def prefill_summary(recs):
    out = {}
    for r in recs:
        if not r.get("t_first"):
            continue
        u = r.get("usage") or {}
        p = u.get("prompt_tokens", 0)
        d = out.setdefault(r["case"], {"prompt": p, "ttft": []})
        d["ttft"].append(r["t_first"] - r["t_send"])
    return {k: {"prompt": d["prompt"], "ttft": st.mean(d["ttft"]), "tps": d["prompt"] / st.mean(d["ttft"])}
            for k, d in out.items()}


def load_steps(t0, t1):
    steps = []
    for p in glob.glob(os.path.join(TRACE, "steps-*.jsonl")):
        if os.path.getmtime(p) < t0:
            continue
        for l in open(p):
            try:
                s = json.loads(l)
            except ValueError:
                continue
            if t0 <= s["ts"] <= t1:
                steps.append(s)
    steps.sort(key=lambda s: s["i"] if "i" in s else s["ts"])
    # GPU time of a step ~ from when it could start (previous step done, or scheduled) to done
    steps.sort(key=lambda s: s["te"])
    prev = None
    for s in steps:
        s["ms"] = 1000.0 * (s["te"] - max(s["ts"], prev["te"] if prev else s["ts"]))
        prev = s
    return steps


def steps_summary(steps):
    out = {}
    for s in steps:
        if s["pt"] == 0:
            key = f"decode x{s['dn']}"
        else:
            b = 256
            while b < s["pt"]:
                b *= 2
            key = f"prefill<={b}" + (f" +{s['dn']}dec" if s["dn"] else " alone")
        out.setdefault(key, []).append(s["ms"])
    return {k: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9), "max": max(v)} for k, v in sorted(out.items())}


def fmt(x, nd=1):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_json, want_steps = "--json" in sys.argv, "--steps" in sys.argv
    allsum = {}
    for trial in args:
        s = {}
        recs, meta = load(trial, "stall")
        if recs:
            s["stall"] = stall_summary(recs)
            if want_steps:
                s["stall_steps"] = steps_summary(load_steps(meta["t0"], meta["saved"]))
        recs, meta = load(trial, "agents")
        if recs:
            s["agents"] = agents_summary(recs, meta)
            if want_steps:
                s["agents_steps"] = steps_summary(load_steps(meta["t0"], meta["t1"]))
        recs, meta = load(trial, "capacity")
        if recs:
            s["capacity"] = capacity_summary(recs, meta)
            if want_steps:
                s["capacity_steps"] = steps_summary(load_steps(meta["t0"], meta["saved"]))
        recs, meta = load(trial, "prefill")
        if recs:
            s["prefill"] = prefill_summary(recs)
        allsum[trial] = s
        if want_json:
            with open(os.path.join(HERE, "results", trial, "summary.json"), "w") as f:
                json.dump(s, f, indent=1)

    # side by side
    T = args
    w = max(14, *(len(t) + 2 for t in T))
    cases = sorted({c for t in T for c in allsum[t].get("stall", {})})
    if cases:
        print("\nSTALL: newcomer TTFT s | keep % | freeze ms   (running streams while a newcomer prefills)")
        print("case".ljust(14) + "".join(t.rjust(w + 10) for t in T))
        for c in cases:
            line = c.ljust(14)
            for t in T:
                d = allsum[t].get("stall", {}).get(c)
                line += (f"{fmt(d['ttft'], 2)} | {fmt(d['keep'], 0)} | {fmt(d['freeze'], 0)}" if d else "-").rjust(w + 10)
            print(line)
    if any("agents" in allsum[t] for t in T):
        print("\nAGENTS")
        keys = ["requests", "errors", "wall_s", "out_tokens", "total_tps", "stream_tps", "ttft_p50", "ttft_p90",
                "ttft_max", "gap_p50", "gap_p99", "gap_max", "gaps05_per_ktok", "gaps1_per_ktok", "stall_pct",
                "tok_per_delivery"]
        print("measure".ljust(16) + "".join(t.rjust(w) for t in T))
        for k in keys:
            print(k.ljust(16) + "".join(fmt(allsum[t].get("agents", {}).get(k), 2).rjust(w) for t in T))
    if any("capacity" in allsum[t] for t in T):
        print("\nCAPACITY: per-stream tok/s (total)")
        cs = sorted({c for t in T for c in allsum[t].get("capacity", {})})
        for c in cs:
            line = f"c={c}".ljust(16)
            for t in T:
                d = allsum[t].get("capacity", {}).get(c)
                line += (f"{fmt(d['stream_tps'])} ({fmt(d['total_tps'], 0)})" if d else "-").rjust(w)
            print(line)
    if any("prefill" in allsum[t] for t in T):
        print("\nPREFILL alone: TTFT s (prompt tok/s)")
        for c in ("fresh10k", "fresh23k", "fresh40k", "fresh66k"):
            line = c.ljust(16)
            for t in T:
                d = allsum[t].get("prefill", {}).get(c)
                line += (f"{fmt(d['ttft'], 2)} ({fmt(d['tps'], 0)})" if d else "-").rjust(w)
            print(line)
    if want_steps:
        for t in T:
            for k in ("stall_steps", "agents_steps", "capacity_steps"):
                if k in allsum[t]:
                    print(f"\nSTEPS {t} {k}: n p50 p90 max (ms)")
                    for kk, d in allsum[t][k].items():
                        print(f"  {kk:28s} {d['n']:6d} {d['p50']:8.1f} {d['p90']:8.1f} {d['max']:8.1f}")


if __name__ == "__main__":
    main()
