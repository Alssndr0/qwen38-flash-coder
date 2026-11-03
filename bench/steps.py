#!/usr/bin/env python3
"""Step-trace analysis: step time by composition, and a cost model fitted on the prefill-carrying steps.
usage: steps.py TRIAL [SCENARIO]      (steps inside that scenario's run window; default: all scenarios of the trial)
       steps.py --since UNIX_TS [--until UNIX_TS]
Cost model (least squares on steps that carry prefill):  ms = a + b * prefill_tokens + c * prefill_tokens * ctx/1e3
                                                              + d * decoding_requests
ctx = context end of the prefill chunk (tokens already in the KV + the chunk)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402


def solve(A, y):
    n = len(A[0])
    M = [[sum(r[i] * r[j] for r in A) for j in range(n)] for i in range(n)]
    v = [sum(r[i] * yy for r, yy in zip(A, y)) for i in range(n)]
    for i in range(n):  # Gauss-Jordan with partial pivoting
        p = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[p], v[i], v[p] = M[p], M[i], v[p], v[i]
        if abs(M[i][i]) < 1e-12:
            continue
        for k in range(n):
            if k != i:
                f = M[k][i] / M[i][i]
                M[k] = [a - f * b for a, b in zip(M[k], M[i])]
                v[k] -= f * v[i]
    return [v[i] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def analyze(steps, label):
    print(f"\n== {label}: {len(steps)} steps")
    for k, d in report.steps_summary(steps).items():
        print(f"  {k:28s} n={d['n']:5d}  p50 {d['p50']:7.1f}  p90 {d['p90']:7.1f}  max {d['max']:7.1f} ms")
    pf = [s for s in steps if s["pt"] > 0 and s["ms"] > 0]
    if len(pf) >= 6:
        A = [[1.0, s["pt"], s["pt"] * max(c for _, c in s["pf"]) / 1e3, s["dn"]] for s in pf]
        y = [s["ms"] for s in pf]
        a, b, c, d = solve(A, y)
        res = [yy - sum(x * w for x, w in zip(r, (a, b, c, d))) for r, yy in zip(A, y)]
        rms = (sum(r * r for r in res) / len(res)) ** 0.5
        print(f"  cost model (n={len(pf)}): ms = {a:.1f} + {b:.4f}*pt + {c:.5f}*pt*ctx/1e3 + {d:.1f}*dn   (rms {rms:.1f} ms)")
        print(f"    => prefill alone at ctx 10k: {1000.0 / (b + c * 10):.0f} tok/s; at ctx 40k: {1000.0 / (b + c * 40):.0f} tok/s")
    dec = [s for s in steps if s["pt"] == 0 and s["ms"] > 0]
    if dec:
        print("  decode-only by streams: " + ", ".join(
            f"x{n}: {report.pct([s['ms'] for s in dec if s['dn'] == n], 0.5):.1f} ms (n={sum(1 for s in dec if s['dn'] == n)})"
            for n in sorted({s["dn"] for s in dec})))


def main():
    a = sys.argv[1:]
    if a and a[0] == "--since":
        t0 = float(a[1])
        t1 = float(a[a.index("--until") + 1]) if "--until" in a else 9e12
        analyze(report.load_steps(t0, t1), f"since {t0}")
        return
    trial = a[0]
    scens = a[1:] or ["prefill", "capacity", "stall", "agents"]
    for sc in scens:
        recs, meta = report.load(trial, sc)
        if not recs:
            continue
        t1 = meta.get("t1") or meta.get("saved")
        analyze(report.load_steps(meta["t0"], t1), f"{trial}/{sc}")


if __name__ == "__main__":
    main()
