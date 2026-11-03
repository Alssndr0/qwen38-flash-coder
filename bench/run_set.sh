#!/usr/bin/env bash
# run the standard scenario set against the serve on :8000 into results/<trial>/
#   run_set.sh TRIAL [quick|full]   quick = stall (2 reps) + capacity + prefill + agents quick
set -uo pipefail
cd "$(dirname "$0")"
T="$1"; MODE="${2:-quick}"
mkdir -p results/$T
{
echo "== $T $(date -u +%FT%TZ) mode $MODE"
python3 conc.py prefill  --trial "$T"
python3 conc.py capacity --trial "$T"
python3 conc.py stall    --trial "$T" --reps 2
python3 conc.py agents   --trial "$T" --set quick
echo "== done $(date -u +%FT%TZ)"
} 2>&1 | tee -a results/$T/run.log
