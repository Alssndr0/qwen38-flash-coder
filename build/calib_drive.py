#!/usr/bin/env python3
"""Drive a calibration serve (MBX_INT8_CALIB_DIR set): arm the recorders (GO file), send the frozen agentic replay
requests as prefill-only calls (max_tokens 1, so every prompt's prefill runs eagerly at >= 64 rows), disarm.
usage: calib_drive.py CALIB_DIR_ON_HOST [--limit N]"""
import argparse, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.expanduser("~/projects/llm-tuning-lab/lib"))
import labkit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("calib_dir"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(labkit.dataset_path("replay_agentic_en_v1.jsonl"))]
if a.limit:
    rows = rows[:a.limit]
go = os.path.join(a.calib_dir, "GO")
open(go, "w").close()
time.sleep(2)
t0 = time.time()
for i, r in enumerate(rows):
    b = dict(r["body"]); b["model"] = os.environ.get("LAB_MODEL", "Qwen/Qwen3.8-Flash-Next"); b["stream"] = False
    for k in ("max_completion_tokens", "stream_options", "store"):
        b.pop(k, None)
    b["max_tokens"] = 1
    req = urllib.request.Request(os.environ.get("LAB_URL", "http://127.0.0.1:8000") + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=600).read()
    except Exception as e:  # noqa: BLE001
        print("request", i, "failed:", repr(e)[:120], flush=True)
    if i % 20 == 0:
        n = len([f for f in os.listdir(a.calib_dir) if f.endswith(".pt") and not f.endswith(".W.pt")])
        print(f"{i + 1}/{len(rows)} requests, {n} row files, {time.time() - t0:.0f} s", flush=True)
os.remove(go)
print("done", flush=True)
