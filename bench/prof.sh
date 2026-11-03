#!/usr/bin/env bash
# Boot trial NAME under nsys (CUDA profiler API capture ranges, one report per range), run prof.py SCEN..., export
# each report to sqlite and summarize it.   prof.sh NAME SCEN [SCEN2 ...]  -> bench/prof/NAME-<n>.{nsys-rep,sqlite,txt}
set -euo pipefail
cd "$(dirname "$0")/.."
N=$1; shift; NS=/opt/nvidia/nsight-systems/2025.3.2; P=$(pwd)/bench/prof
mkdir -p "$P"; rm -f "$P"/$N-*.nsys-rep
python3 bench/mkrecipe.py prof-$N --base bench/trials/$N.yaml "vllm.profiler-config={\"profiler\":\"cuda\"}" vllm.gpu-memory-utilization=0.85 >/dev/null
EXTRA_DOCKER="-v $NS:$NS:ro -v $P:/prof" \
WRAP="$NS/target-linux-sbsa-armv8/nsys profile -o /prof/$N-%p -t cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=repeat:${#} --cuda-graph-trace=node --trace-fork-before-exec=true --force-overwrite=true" \
  bench/up.sh prof-$N
python3 bench/prof.py "$@" | tee "$P/$N.req.txt"
sleep 30   # let nsys write the reports
docker stop -t 90 qwen38-flash-next >/dev/null || true
sleep 5
for REP in $(ls -tr "$P"/$N-*.nsys-rep); do
  b=${REP%.nsys-rep}
  "$NS/bin/nsys" export --type sqlite --force-overwrite true -o "$b.sqlite" "$REP" >/dev/null 2>&1 || { echo "export failed: $REP"; continue; }
  python3 ~/projects/qwen38-spark-opt/scripts/nsys_analyze.py "$b.sqlite" --top 50 > "$b.txt" || true
  echo "== $b"; head -25 "$b.txt"
done
