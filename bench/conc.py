#!/usr/bin/env python3
"""Concurrency and smoothness bench for the coder serve (stdlib only).

Every request streams with per-chunk token ids, so each SSE event is one engine step's worth of tokens for that
request; every event is stored with its wall-clock time (time.time, the same clock as the serve's step tracer).

Scenarios (results in bench/results/<trial>/<scenario>.jsonl, one JSON line per request, plus <scenario>.meta.json):
  stall     B requests keep decoding (real OpenCode prompts); after a steady spell, one more request arrives with a
            prompt that has to be prefilled. Cases: fresh:<row> (nothing cached) and delta:<rowA>><rowB> (row A's
            prompt cached first, then row B, the same session a few turns later). Measures what the decoding streams
            see while the newcomer prefills, and the newcomer's time to first token.
  agents    W workers replay recorded OpenCode / Pi sessions turn by turn (closed loop, think time between turns),
            the realistic 4-agent load. Per request: TTFT, decode speed, gaps.
  capacity  c = 1..C streams decoding with nothing arriving: per-stream speed at each concurrency.
  prefill   one fresh prompt alone: time to first token = prefill speed.

usage:
  conc.py stall    --trial T [--bg 3] [--cases fresh:big,delta:...] [--reps 2]
  conc.py agents   --trial T [--workers 4] [--set quick|full] [--think 1.0]
  conc.py capacity --trial T [--max-c 4] [--secs 15]
  conc.py prefill  --trial T
Then: report.py T [T2 ...]
"""
import argparse
import copy
import http.client
import json
import os
import random
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CONC_DATA", os.path.expanduser("~/projects/llm-tuning-lab/datasets/data/replay_agentic_en_v1.jsonl"))
MODEL = os.environ.get("CONC_MODEL", "Qwen/Qwen3.8-Flash-Next")
HOST, PORT = os.environ.get("CONC_HOST", "127.0.0.1"), int(os.environ.get("CONC_PORT", "8000"))


# ---------------------------------------------------------------------------------------------- data
def load_rows():
    rows = [json.loads(l) for l in open(DATA)]
    for r in rows:
        r["orig_prompt_tokens"] = int(r["orig_prompt_tokens"])
        r["orig_completion_tokens"] = int(r["orig_completion_tokens"])
        r["turn"] = int(r["turn"])
        r["seed"] = int(r["seed"])
    return rows


def sessions(rows):
    s = {}
    for r in rows:
        s.setdefault(r["session"], []).append(r)
    for v in s.values():
        v.sort(key=lambda r: r["turn"])
    return s


def body_of(row, max_tokens, min_tokens=0, nonce=None):
    b = copy.deepcopy(row["body"])
    b["model"] = MODEL
    b["stream"] = True
    b["stream_options"] = {"include_usage": True}
    b["return_token_ids"] = True
    for k in ("max_completion_tokens", "max_tokens", "store"):
        b.pop(k, None)
    b["max_tokens"] = max_tokens
    if min_tokens:
        b["min_tokens"] = min_tokens
    b["seed"] = row["seed"]
    if nonce:
        m0 = b["messages"][0]
        if isinstance(m0.get("content"), str):
            m0["content"] = f"[session {nonce}]\n" + m0["content"]
        else:
            b["messages"].insert(0, {"role": "system", "content": f"[session {nonce}]"})
    return b


# ---------------------------------------------------------------------------------------------- client
def stream(body, rec, stop=None, timeout=3600):
    """POST a streaming chat completion; fill rec with t_send, t_first, ev=[[t, ntok, kind]], usage, finish."""
    rec.update(t_send=time.time(), t_first=None, ev=[], usage=None, finish=None, err=None)
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            rec["err"] = f"HTTP {r.status}: {r.read()[:300]!r}"
            return rec
        for raw in r:
            if stop is not None and stop.is_set():
                rec["finish"] = "stopped"
                break
            if not raw.startswith(b"data: ") or raw.startswith(b"data: [DONE]"):
                continue
            t = time.time()
            d = json.loads(raw[6:])
            if d.get("usage") and not d.get("choices"):
                rec["usage"] = d["usage"]
                continue
            for ch in d.get("choices") or []:
                n = len(ch.get("token_ids") or [])
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
                if not n:
                    continue
                dl = ch.get("delta") or {}
                kind = "t" if dl.get("tool_calls") else ("r" if (dl.get("reasoning_content") or dl.get("reasoning")) else "c")
                if rec["t_first"] is None:
                    rec["t_first"] = t
                rec["ev"].append([round(t, 4), n, kind])
        c.close()
    except Exception as e:  # keep the record, mark it
        rec["err"] = repr(e)
    rec["t_end"] = time.time()
    return rec


def start(body, rec, stop=None):
    th = threading.Thread(target=stream, args=(body, rec, stop), daemon=True)
    th.start()
    return th


def wait_first(recs, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(r.get("t_first") or r.get("err") for r in recs):
            return True
        time.sleep(0.05)
    return False


def health():
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=5)
        c.request("GET", "/v1/models")
        return c.getresponse().status == 200
    except Exception:
        return False


def outdir(trial):
    d = os.path.join(HERE, "results", trial)
    os.makedirs(d, exist_ok=True)
    return d


def save(trial, scen, recs, meta):
    d = outdir(trial)
    with open(os.path.join(d, f"{scen}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    meta["saved"] = time.time()
    with open(os.path.join(d, f"{scen}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


# ---------------------------------------------------------------------------------------------- scenarios
# background decoders: three OpenCode sessions mid-way (prompts 12-29k tokens: typical agent contexts)
BG_IDS = ["b1.opencode.swe-django-django-14999#20", "b1.opencode.swe-sympy-sympy-24443#4",
          "b1.opencode.poly-rust-wordy#4", "b1.opencode.local-opt-refactor#5",
          "b1.opencode.swe-pydata-xarray-4356#5", "b1.opencode.poly-go-dominoes#5"]
# arrivals: fresh = nothing cached (a new session with a big context); delta = same session, later turn
CASES = {
    "fresh5k": ("fresh", "b1.opencode.local-dash-tests#0"),             # 5.2k
    "fresh10k": ("fresh", "b1.opencode.local-opt-refactor#3"),          # 9.8k
    "fresh23k": ("fresh", "b1.opencode.local-vllm-rejection#5"),        # 22.6k
    "fresh40k": ("fresh", "b1.opencode.swe-django-django-11848#23"),     # 40.4k
    "fresh66k": ("fresh", "b1.opencode.swe-sympy-sympy-13031#32"),       # 65.6k
    "delta3k@40k": ("delta", "b1.opencode.swe-django-django-11848#23", "b1.opencode.swe-django-django-11848#27"),
    "delta12k@5k": ("delta", "b1.opencode.local-recipe-explain#0", "b1.opencode.local-recipe-explain#1"),
    "pi2to3": ("delta", "b1.pi.swe-django-django-14999#2", "b1.pi.swe-django-django-14999#3"),
    "pi3to4": ("delta", "b1.pi.swe-django-django-14999#3", "b1.pi.swe-django-django-14999#4"),
    "delta2k@66k": ("delta", "b1.opencode.swe-sympy-sympy-13031#32", "b1.opencode.swe-sympy-sympy-13031#35"),
}


def cmd_stall(a):
    rows = {r["id"]: r for r in load_rows()}
    cases = a.cases.split(",")
    out, meta = [], {"scenario": "stall", "bg": a.bg, "cases": cases, "reps": a.reps, "t0": time.time()}
    for rep in range(a.reps):
        for case in cases:
            spec = CASES[case]
            # a fresh cache key per case and rep: fresh = nothing cached; delta = only row A's prompt cached
            nonce = uuid.uuid4().hex
            if spec[0] == "delta":  # warm the cached part (not measured)
                warm = {}
                stream(body_of(rows[spec[1]], 1, nonce=nonce), warm)
            stop = threading.Event()
            bg = []
            for i in range(a.bg):
                rec = {"role": "bg", "case": case, "rep": rep, "row": BG_IDS[i]}
                bg.append(rec)
                start(body_of(rows[BG_IDS[i]], a.bg_tokens, a.bg_tokens), rec, stop)
            wait_first(bg)
            time.sleep(a.settle)
            inj = {"role": "inject", "case": case, "rep": rep, "row": spec[-1]}
            stream(body_of(rows[spec[-1]], a.inject_tokens, a.inject_tokens, nonce=nonce), inj)
            time.sleep(a.tail)
            stop.set()
            t_stop = time.time()
            while any(r.get("t_end") is None and not r.get("err") for r in bg) and time.time() - t_stop < 30:
                time.sleep(0.05)
            out.extend(bg + [inj])
            ttft = (inj["t_first"] or inj["t_end"]) - inj["t_send"]
            print(f"  {case} rep {rep}: newcomer TTFT {ttft:.2f}s", flush=True)
            time.sleep(1.0)
    save(a.trial, "stall", out, meta)


def agent_sets(rows):
    s = sessions(rows)
    quick = ["b1.opencode.swe-django-django-14999", "b1.opencode.local-vllm-rejection", "b1.opencode.poly-rust-wordy",
             "b1.opencode.swe-pydata-xarray-4356", "b1.opencode.swe-sympy-sympy-24443", "b1.pi.swe-django-django-14999",
             "b1.opencode.local-dash-tests", "b1.pi.swe-sympy-sympy-13031"]
    return {"quick": [s[k] for k in quick], "full": [s[k] for k in sorted(s)]}


def cmd_agents(a):
    rows = load_rows()
    sets = agent_sets(rows)
    queue = list(sets[a.set])
    random.Random(7).shuffle(queue)
    lock = threading.Lock()
    out = []
    meta = {"scenario": "agents", "workers": a.workers, "set": a.set, "think": a.think, "cap": a.cap, "t0": time.time()}

    def worker(w):
        while True:
            with lock:
                if not queue:
                    return
                sess = queue.pop(0)
            for row in sess:
                cap = min(a.cap, 2 * row["orig_completion_tokens"] + 64)
                rec = {"role": "agent", "worker": w, "row": row["id"], "harness": row["harness"],
                       "orig_prompt": row["orig_prompt_tokens"], "cap": cap}
                stream(body_of(row, cap), rec)
                with lock:
                    out.append(rec)
                    n = len(out)
                u = rec.get("usage") or {}
                cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                print(f"  [{n:3d}] w{w} {row['id'][:48]:48s} prompt {u.get('prompt_tokens', 0):6d} "
                      f"cached {cached:6d} out {u.get('completion_tokens', 0):5d} "
                      f"ttft {((rec['t_first'] or rec['t_end']) - rec['t_send']):5.2f}s {rec.get('err') or ''}",
                      flush=True)
                time.sleep(a.think)

    ths = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(a.workers)]
    # stagger the start a little, as real agents do
    for i, th in enumerate(ths):
        th.start()
        time.sleep(a.stagger)
    for th in ths:
        th.join()
    meta["t1"] = time.time()
    save(a.trial, "agents", out, meta)


def cmd_capacity(a):
    rows = {r["id"]: r for r in load_rows()}
    out, meta = [], {"scenario": "capacity", "max_c": a.max_c, "secs": a.secs, "t0": time.time()}
    for c in range(1, a.max_c + 1):
        stop = threading.Event()
        recs = []
        for i in range(c):
            rec = {"role": "cap", "c": c, "row": BG_IDS[i]}
            recs.append(rec)
            start(body_of(rows[BG_IDS[i]], 8000, 8000), rec, stop)
        wait_first(recs)
        time.sleep(a.secs + 2)
        stop.set()
        t_stop = time.time()
        while any(r.get("t_end") is None and not r.get("err") for r in recs) and time.time() - t_stop < 30:
            time.sleep(0.05)
        out.extend(recs)
        print(f"  c={c} done", flush=True)
        time.sleep(1.0)
    save(a.trial, "capacity", out, meta)


def cmd_prefill(a):
    rows = {r["id"]: r for r in load_rows()}
    out, meta = [], {"scenario": "prefill", "t0": time.time()}
    for rep in range(a.reps):
        for case in ("fresh10k", "fresh23k", "fresh40k", "fresh66k"):
            rec = {"role": "prefill", "case": case, "rep": rep, "row": CASES[case][1]}
            stream(body_of(rows[CASES[case][1]], 16, 16, nonce=uuid.uuid4().hex), rec)
            out.append(rec)
            print(f"  {case}: TTFT {(rec['t_first'] or rec['t_end']) - rec['t_send']:.2f}s", flush=True)
    save(a.trial, "prefill", out, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stall")
    p.add_argument("--trial", required=True)
    p.add_argument("--bg", type=int, default=3)
    p.add_argument("--bg-tokens", type=int, default=6000)
    p.add_argument("--inject-tokens", type=int, default=48)
    p.add_argument("--settle", type=float, default=4.0)
    p.add_argument("--tail", type=float, default=2.0)
    p.add_argument("--cases", default="fresh10k,fresh23k,fresh40k,delta3k@40k,delta12k@5k")
    p.add_argument("--reps", type=int, default=2)
    p = sp.add_parser("agents")
    p.add_argument("--trial", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--set", default="quick")
    p.add_argument("--think", type=float, default=1.0)
    p.add_argument("--stagger", type=float, default=2.0)
    p.add_argument("--cap", type=int, default=2048)
    p = sp.add_parser("capacity")
    p.add_argument("--trial", required=True)
    p.add_argument("--max-c", type=int, default=4)
    p.add_argument("--secs", type=float, default=15.0)
    p = sp.add_parser("prefill")
    p.add_argument("--trial", required=True)
    p.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    if not health():
        sys.exit(f"serve not answering on {HOST}:{PORT}")
    {"stall": cmd_stall, "agents": cmd_agents, "capacity": cmd_capacity, "prefill": cmd_prefill}[a.cmd](a)


if __name__ == "__main__":
    main()
