#!/usr/bin/env python3
"""Robustness checks for a scheduling mode (run against the live serve): each prints PASS/FAIL.
  abort:   3 streams decode; a 23k newcomer is cut off by the client mid-prefill (several times); serve still answers.
  long:    3 streams decode; a 66k fresh prompt prefills (many chunks); its answer starts, serve still answers.
  burst:   8 newcomers at once (mixed sizes) while 2 stream; all complete.
  logprob: 2 streams decode; a request with logprobs arrives; it completes (not starved).
  image-free requests only."""
import http.client, json, os, sys, threading, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conc
rows = {r["id"]: r for r in conc.load_rows()}

def bg(n, stop):
    recs = []
    for i in range(n):
        r = {}; recs.append(r)
        conc.start(conc.body_of(rows[conc.BG_IDS[i]], 4000, 4000), r, stop)
    conc.wait_first(recs)
    return recs

def ok_health():
    b = {"model": conc.MODEL, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 8,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = {}; conc.stream(dict(b, stream=True, return_token_ids=True), r)
    return not r.get("err") and r.get("t_first") is not None

def t_abort():
    stop = threading.Event(); b = bg(3, stop); time.sleep(1)
    for k in range(3):
        st = threading.Event(); rec = {}
        th = conc.start(conc.body_of(rows[conc.CASES["fresh23k"][1]], 32, 32, nonce=uuid.uuid4().hex), rec, st)
        time.sleep(1.5 + k); st.set(); th.join(30)
    time.sleep(2); alive = all(not r.get("err") for r in b); stop.set(); time.sleep(1)
    return alive and ok_health()

def t_long():
    stop = threading.Event(); b = bg(3, stop); rec = {}
    conc.stream(conc.body_of(rows[conc.CASES["fresh66k"][1]], 32, 32, nonce=uuid.uuid4().hex), rec)
    stop.set(); time.sleep(1)
    print(f"    66k TTFT {(rec.get('t_first') or 0) - rec['t_send']:.1f}s tokens {sum(n for _, n, _ in rec.get('ev', []))}")
    return not rec.get("err") and rec.get("t_first") and ok_health()

def t_burst():
    stop = threading.Event(); b = bg(2, stop)
    ids = ["b1.opencode.local-dash-tests#2", "b1.opencode.poly-go-dominoes#3", "b1.pi.swe-django-django-14999#4",
           "b1.opencode.swe-sympy-sympy-24443#6", "b1.opencode.poly-rust-wordy#6", "b1.opencode.local-recipe-explain#1",
           "b1.pi.swe-sympy-sympy-13031#18", "b1.opencode.swe-pydata-xarray-4356#3"]
    recs = [{} for _ in ids]
    ths = [conc.start(conc.body_of(rows[i], 64, 64, nonce=uuid.uuid4().hex), r) for i, r in zip(ids, recs)]
    for t in ths: t.join(600)
    stop.set(); time.sleep(1)
    print("    burst TTFTs " + " ".join(f"{(r.get('t_first') or 0) - r['t_send']:.1f}" for r in recs))
    return all(not r.get("err") and sum(n for _, n, _ in r.get("ev", [])) == 64 for r in recs) and ok_health()

def t_logprob():
    stop = threading.Event(); b = bg(2, stop); time.sleep(1)
    body = conc.body_of(rows["b1.opencode.local-dash-tests#3"], 32, 32, nonce=uuid.uuid4().hex)
    body["logprobs"] = True; body["top_logprobs"] = 2
    rec = {}; t0 = time.time(); conc.stream(body, rec)
    stop.set(); time.sleep(1)
    print(f"    logprobs request TTFT {(rec.get('t_first') or 0) - rec['t_send']:.1f}s")
    return not rec.get("err") and rec.get("t_first") is not None

for name in (sys.argv[1:] or ["abort", "long", "burst", "logprob"]):
    t0 = time.time()
    try:
        res = globals()["t_" + name]()
    except Exception as e:
        res = False; print("   ", repr(e))
    print(f"  {name:8s} {'PASS' if res else 'FAIL'}  ({time.time() - t0:.0f} s)", flush=True)
