#!/usr/bin/env bash
# Record of the 2026-11-19 trial (base34 then bp34). It booted with files/mounts-bp34.txt = files/mounts.txt + the five
# files/v34 files then (incl. a qwen4_exp model.py copy and gpu_worker.py with the post-capture warm-up, #62560).
# Both were removed after the decision (bench/DECISIONS.md, 2026-11-19 09:40); to rerun, recreate that list.
# The 0.34 backport trial: base34 then bp34, each freshly booted and gated. Log: logs/bp34.log
cd "$(dirname "$0")"
python3 mkrecipe.py base34 > /dev/null && python3 mkrecipe.py bp34 > /dev/null
./up.sh base34 || { echo "== base34 boot FAILED"; exit 1; }
./bp34_gates.sh base34
MOUNTS=files/mounts-bp34.txt ./up.sh bp34 || { echo "== bp34 boot FAILED"; exit 1; }
./bp34_gates.sh bp34
echo "== $(date -u +%T) compare"
python3 exact_boot.py compare base34-a0 base34-a1 bp34-a0 bp34-a1 base34-a0 bp34-a0 base34-a1 bp34-a1
echo "== all done $(date -u +%T)"
