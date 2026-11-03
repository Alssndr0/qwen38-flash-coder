#!/usr/bin/env python3
"""Live sweep of the pacing policy on the stall scenario (serve booted with mbx_sched + tracer).
For each policy: write cache/mbx-smooth.json, wait for the serve to pick it up, run the stall cases once.
usage: sweep_cap.py TRIALPREFIX POLICY_JSON [POLICY_JSON ...]   (each policy JSON gets a short name via "name")
Results: results/<TRIALPREFIX>-<name>/stall.jsonl"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "..", "cache", "mbx-smooth.json")
prefix = sys.argv[1]
cases = os.environ.get("CASES", "fresh10k,fresh23k,delta3k@40k,delta12k@5k")
reps = os.environ.get("REPS", "1")
for pj in sys.argv[2:]:
    p = json.loads(pj)
    name = p.pop("name")
    p["trace"] = True
    with open(LIVE + ".tmp", "w") as f:
        json.dump(p, f)
    os.replace(LIVE + ".tmp", LIVE)
    time.sleep(1.5)
    print(f"== {prefix}-{name}: {p}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "conc.py"), "stall", "--trial", f"{prefix}-{name}",
                    "--cases", cases, "--reps", reps], check=False)
