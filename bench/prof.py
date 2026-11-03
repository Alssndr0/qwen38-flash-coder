#!/usr/bin/env python3
"""Profile one scenario on a serve booted under nsys (prof.sh): warm up, POST /start_profile, run, POST /stop_profile.
usage: prof.py prefill10k | mixed10k | decode4"""
import http.client, json, sys, threading, time, uuid
sys.path.insert(0, __import__("os").path.dirname(__file__))
import conc

whats = sys.argv[1:]
rows = {r["id"]: r for r in conc.load_rows()}


def post(p):
    c = http.client.HTTPConnection(conc.HOST, conc.PORT, timeout=600)
    c.request("POST", p, b"")
    return c.getresponse().read()


fresh = rows[conc.CASES["fresh10k"][1]]
w = {}
conc.stream(conc.body_of(fresh, 8, 8, nonce=uuid.uuid4().hex), w)  # warm-up
for what in whats:
  print("==", what, flush=True)
  if what == "prefill10k":
      post("/start_profile"); time.sleep(0.5)
      r = {}
      conc.stream(conc.body_of(fresh, 1, 1, nonce=uuid.uuid4().hex), r)
      time.sleep(0.5); post("/stop_profile")
      print("ttft", r["t_first"] - r["t_send"], r.get("usage"))
  elif what in ("mixed10k", "decode4"):
      stop = threading.Event()
      bg = [{} for _ in range(3 if what == "mixed10k" else 4)]
      for i, b in enumerate(bg):
          conc.start(conc.body_of(rows[conc.BG_IDS[i]], 4000, 4000), b, stop)
      conc.wait_first(bg); time.sleep(3)
      post("/start_profile"); time.sleep(0.3)
      if what == "mixed10k":
          r = {}
          conc.stream(conc.body_of(fresh, 1, 1, nonce=uuid.uuid4().hex), r)
          print("ttft", r["t_first"] - r["t_send"])
      else:
          time.sleep(3)
      time.sleep(0.3); post("/stop_profile"); stop.set(); time.sleep(2)
