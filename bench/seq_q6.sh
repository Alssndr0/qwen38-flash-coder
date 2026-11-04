#!/usr/bin/env bash
# The sequence that preceded the 10:15 crash, with memory logged per second, on the running serve. usage: seq_q6.sh NAME
cd "$(dirname "$0")"; N=$1; LAB=~/projects/llm-tuning-lab
export LAB_URL=http://127.0.0.1:8000 LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
: > logs/mem-$N.log; ./memwatch.sh logs/mem-$N.log 3600 &
W=$!
st() { echo "== $(date -u +%T) $N $*: mem $(tail -1 logs/mem-$N.log)"; }
[ "${2:-}" = sampled ] && { st Q6-sampled; python3 $LAB/tests/quality/codeeval.py "${3}-s" --sampling '{"temperature":1.0,"top_p":0.95,"top_k":20}' --samples 2 2>&1 | tail -2 | cut -c1-120; }
st run_set; ./run_set.sh "$N" quick > /dev/null 2>&1
st P3; python3 $LAB/tests/perf/nextturn.py "$N" --model "$LAB_MODEL" --url "$LAB_URL" 2>&1 | tail -1 | cut -c1-120
st Q6; python3 $LAB/tests/quality/codeeval.py "$N" 2>&1 | tail -2 | cut -c1-120
st end; echo "== serve up: $(docker ps -q --filter name=qwen38-flash-next --filter status=running | wc -l); min $(sort -k2 -n logs/mem-$N.log | awk '$4==1' | head -1)"
kill $W 2>/dev/null; echo "== $N done"
