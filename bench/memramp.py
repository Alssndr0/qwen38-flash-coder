#!/usr/bin/env python3
"""Worker GPU memory while load ramps up (vLLM #62560 check): nvidia-smi used memory of VLLM::Worker after boot, then
during and after 1, 2, 4, 8, 16 concurrent streams (the serve's real sampling: T 1, top-p 0.95, top-k 20; fixed-length
decode), the whole ramp twice. Growth that the second ramp does not repeat is the allocator caching new batch sizes.
  memramp.py NAME [--tokens 384] -> results/memramp/NAME.json + one summary line
"""
import argparse
import json
import os
import subprocess
import threading
import time
import uuid

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("LAB_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen3.8-Flash-Next")


def worker_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(l.split(",")[1]) for l in out.strip().split("\n") if "VLLM::Worker" in l)


def stream(i, tokens, errs):
    salt = uuid.uuid4().hex[:12]
    body = {"model": MODEL, "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                          f"[{salt}] Write a Python module for task {i}: a small job queue with retries, a priority "
                          f"heap, persistence to SQLite and unit tests. Explain each design choice as you go."}]}
    try:
        r = requests.post(URL + "/v1/chat/completions", json=body, timeout=900)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e)[:200])


def level(c, tokens):
    errs, peak = [], [worker_mib()]
    ts = [threading.Thread(target=stream, args=(i, tokens, errs)) for i in range(c)]
    t0 = time.time()
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        peak.append(worker_mib())
        time.sleep(0.25)
    time.sleep(1.0)
    return {"c": c, "s": round(time.time() - t0, 1), "peak_mib": max(peak), "after_mib": worker_mib(), "errors": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--tokens", type=int, default=384)
    a = ap.parse_args()
    res = {"name": a.name, "boot_mib": worker_mib(), "ramps": []}
    for _ in range(2):
        res["ramps"].append([level(c, a.tokens) for c in (1, 2, 4, 8, 16)])
    res["end_mib"] = worker_mib()
    os.makedirs(os.path.join(HERE, "results", "memramp"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "memramp", a.name + ".json"), "w"), indent=1)
    r1, r2 = res["ramps"]
    steps = " ".join(f"c{x['c']}:{x['after_mib'] - res['boot_mib']:+d}" for x in r1)
    print(f"{a.name}: boot {res['boot_mib']} MiB | ramp1 after each level vs boot {steps} | "
          f"ramp1 peak {max(x['peak_mib'] for x in r1)} | ramp2 growth {r2[-1]['after_mib'] - r1[-1]['after_mib']:+d} | "
          f"end {res['end_mib']} ({res['end_mib'] - res['boot_mib']:+d}) | errors {sum(len(x['errors']) for x in r1 + r2)}")


if __name__ == "__main__":
    main()
