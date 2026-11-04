#!/usr/bin/env bash
# Gates for the vLLM 0.33 backport against the serve on :8000. usage: bp33_gates.sh NAME  (results/, results/calib, logs/bp33-NAME.log)
set -uo pipefail
cd "$(dirname "$0")"; N=$1; LAB=~/projects/llm-tuning-lab
export LAB_URL=http://127.0.0.1:8000 LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
step() { echo "== $(date -u +%T) $N $*"; }
step calib;     python3 calib.py "$N"
step exact;     python3 exact_boot.py save "$N-a0"; python3 exact_boot.py save "$N-a1"
step run_set;   ./run_set.sh "$N" quick > /dev/null 2>&1; tail -1 "results/$N/run.log"
step P3;        python3 $LAB/tests/perf/nextturn.py "$N" --model "$LAB_MODEL" --url "$LAB_URL" | tail -1
step Q6;        python3 $LAB/tests/quality/codeeval.py "$N" | tail -2
step Q6-sampled; python3 $LAB/tests/quality/codeeval.py "$N-s" --sampling '{"temperature":1.0,"top_p":0.95,"top_k":20}' --samples 2 | tail -2
step done
