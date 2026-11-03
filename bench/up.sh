#!/usr/bin/env bash
# Boot trial recipe bench/trials/NAME.yaml on :8000 through the kit's run.sh; returns when healthy (warm-up done).
#   up.sh NAME            (EXTRA_DOCKER / WRAP pass through, e.g. for nsys)
# Log: bench/logs/boot-NAME-HHMMSS.log
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; R=bench/trials/$N.yaml
[ -f "$R" ] || { echo "no $R"; exit 1; }
mkdir -p bench/logs
LOG=bench/logs/boot-$N-$(date +%H%M%S).log
echo "$(date +%F_%T) boot $N -> $LOG"
t0=$(date +%s)
if RECIPE="$R" ./run.sh > "$LOG" 2>&1 < /dev/null; then
  echo "$(date +%F_%T) $N healthy after $(( $(date +%s) - t0 )) s; $(grep -m1 'warm-up' "$LOG" || true)"
  echo "$N" > bench/logs/current
else
  echo "$(date +%F_%T) $N FAILED"; docker logs qwen38-flash-next > "${LOG%.log}-container.log" 2>&1 || true
  grep -E "Error|error|Traceback" "${LOG%.log}-container.log" | tail -15; exit 1
fi
