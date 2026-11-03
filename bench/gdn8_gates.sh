#!/usr/bin/env bash
# C1 gates on the running gdn8 boot (pre-registered in DECISIONS.md, 2026-11-03 03:20): quality with the MXFP8 prefill
# path on, then speed paired live (off / on). Quality results go next to the v4gdnq gate results they are compared to
# (~/projects/fp4local-az-eval/suite); speed results to bench/results/.
#   bench/gdn8_gates.sh [q1] [q6] [p1] [speed]      (default: all)
set -uo pipefail
cd "$(dirname "$0")"
B=$(pwd); G=$B/../cache/mbx-gdn8.json; S=$B/../cache/mbx-smooth.json
T="${*:-q1 q6 p1 speed}"
export LAB=~/projects/llm-tuning-lab LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
SUITE=~/projects/fp4local-az-eval/suite
on()  { echo '{"min_tokens": 256}' > "$G"; sleep 3; }
off() { echo '{"min_tokens": 100000000}' > "$G"; sleep 3; }
pol() {  # smooth policy with the slice cost model's per-token term
  python3 - "$S" "$1" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); p["lb"] = float(sys.argv[2]); json.dump(p, open(sys.argv[1], "w"))
PY
  sleep 1.5
}
cd "$SUITE"
if [[ " $T " == *" q1 "* ]]; then
  for v in on off; do
    $v; echo "$(date -u +%T) Q1 capture gdn8 $v"
    python3 $LAB/tests/quality/klprobe.py capture results/kl-gdn8-$v.json
  done
  for r in kl-v4gdnq kl-f5c-final kl-gdn8-off; do
    echo "-- gdn8-on vs $r"; python3 $LAB/tests/quality/klprobe.py compare results/$r.json results/kl-gdn8-on.json | tail -2
  done
  echo "-- gdn8-off vs kl-v4gdnq (floor)"; python3 $LAB/tests/quality/klprobe.py compare results/kl-v4gdnq.json results/kl-gdn8-off.json | tail -2
fi
if [[ " $T " == *" q6 "* ]]; then
  on; echo "$(date -u +%T) Q6 gdn8 on"
  python3 $LAB/tests/quality/codeeval.py gdn8 | tail -4
  python3 $LAB/tests/quality/codeeval.py --compare v4gdnq gdn8 | tail -6
fi
if [[ " $T " == *" p1 "* ]]; then
  echo "$(date -u +%T) P1 quick (Q2 CJK, Q3 tool args), off / on alternating per chunk"
  mkdir -p "$B/logs"; V="$B/logs/gdn8-variants.json"
  echo '[{"name": "w4", "live": {"min_tokens": 100000000}}, {"name": "g8", "live": {"min_tokens": 256}}]' > "$V"
  python3 $LAB/tests/perf/replay.py --run p1q-gdn8 --data replay_agentic_en_v1.jsonl --live "$V" --live-file "$G" \
    --max-tokens 2048 --limit 28 --note "C1 gdn8 quick, paired live" | grep -v '^chunk' | tail -12
fi
cd "$B"
if [[ " $T " == *" speed "* ]]; then
  for v in off on; do
    $v; if [ $v = on ]; then pol 0.0067; else pol 0.0071; fi
    echo "$(date -u +%T) stall + agents, gdn8 $v"
    python3 conc.py stall --trial g8s-$v --reps 2 | tail -12
    python3 conc.py agents --trial g8a-$v --set quick | tail -3
  done
  off; pol 0.0071
  python3 report.py g8s-off g8s-on | tail -30
  python3 report.py g8a-off g8a-on | tail -30
fi
off  # between tests the serve behaves exactly like v4gdnq
echo "$(date -u +%T) gates done"
