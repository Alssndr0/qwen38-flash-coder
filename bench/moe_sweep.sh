#!/usr/bin/env bash
# Marlin MoE row tile on prefill-size batches, live (files/mbx_moe.py; boot with env MBX_MOE_LIVE=/cache/mbx-moe.json):
# prefill alone (fresh 10k / 23k / 40k / 66k) per tile, then the report side by side.
#   bench/moe_sweep.sh [TAG] [BM ...]      (default: 0 = vLLM's choice, 16, 32, 48, 64)
set -uo pipefail
cd "$(dirname "$0")"
TAG=${1:-ms}; shift || true
BMS="${*:-0 16 32 48 64}"
L=../cache/mbx-moe.json
ts=()
for bm in $BMS; do
  echo "{\"bm\": $bm, \"min_tokens\": 256}" > "$L"; sleep 3
  echo "$(date -u +%T) bm $bm"
  python3 conc.py prefill --trial $TAG-bm$bm --reps 1 | tail -4
  ts+=($TAG-bm$bm)
done
echo '{"bm": 0}' > "$L"
python3 report.py "${ts[@]}" | sed -n '/PREFILL/,$p'
