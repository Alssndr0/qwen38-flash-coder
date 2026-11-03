#!/usr/bin/env python3
"""Write a trial recipe: recipe.yaml with keys set or removed, line by line (serve.sh reads a two-level YAML).
usage: mkrecipe.py NAME [--base recipe.yaml] [section.key=value ...] [-section.key ...]
  e.g. mkrecipe.py tr-base vllm.scheduler-cls=mbx_sched.SmoothScheduler env.MBX_STEP_TRACE=/cache/steptrace
  -> bench/trials/NAME.yaml (each changed line marked # [trial])"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
name = args.pop(0)
base = os.path.join(HERE, "..", "recipe.yaml")
if "--base" in args:
    i = args.index("--base"); base = args[i + 1]; del args[i:i + 2]
lines = open(base).read().split("\n")
for a in args:
    drop = a.startswith("-")
    a = a.lstrip("-")
    key, _, val = a.partition("=")
    sec, _, k = key.partition(".")
    # find the section
    si = next(i for i, l in enumerate(lines) if re.match(rf"^{re.escape(sec)}:\s*$", l))
    ei = next((i for i in range(si + 1, len(lines)) if re.match(r"^[A-Za-z_]", lines[i])), len(lines))
    ki = next((i for i in range(si + 1, ei) if re.match(rf"^\s+{re.escape(k)}:(\s|$)", lines[i])), None)
    if val and not re.match(r"^[-\w./:]+$", val):
        val = "'" + val + "'" if "'" not in val else val
    new = f"  {k}: {val}   # [trial]"
    if drop:
        if ki is not None:
            lines[ki] = f"  # {k}: removed   # [trial]"
    elif ki is not None:
        lines[ki] = new
    else:
        lines.insert(si + 1, new)
os.makedirs(os.path.join(HERE, "trials"), exist_ok=True)
out = os.path.join(HERE, "trials", name + ".yaml")
open(out, "w").write("\n".join(lines))
print(out)
