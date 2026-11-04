#!/usr/bin/env bash
# A/B for the 16-stream memory failure: fresh boot of trial T, memory logged per second, then the 16-concurrent code eval.
# usage: ab_q6.sh T LABEL
cd "$(dirname "$0")"; T=$1; N=$2; LAB=~/projects/llm-tuning-lab
export LAB_URL=http://127.0.0.1:8000 LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
: > logs/mem-$N.log; ./memwatch.sh logs/mem-$N.log 2400 &
W=$!
./up.sh "$T"
echo "== $(date -u +%T) after boot: $(tail -1 logs/mem-$N.log)"
python3 $LAB/tests/quality/codeeval.py "$N" --sets humaneval,mbpp 2>&1 | tail -3 | cut -c1-140
echo "== $(date -u +%T) serve up: $(docker ps -q --filter name=qwen38-flash-next --filter status=running | wc -l); min MemAvailable $(sort -k2 -n logs/mem-$N.log | awk '$4==1' | head -1)"
kill $W 2>/dev/null
echo "== $N done"
