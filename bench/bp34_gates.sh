#!/usr/bin/env bash
# Gates for the vLLM 0.34 backport against the serve on :8000, right after its boot. usage: bp34_gates.sh NAME
set -uo pipefail
cd "$(dirname "$0")"; N=$1
export LAB_URL=http://127.0.0.1:8000 LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
step() { echo "== $(date -u +%T) $N $*"; }
mib() { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits | awk -F', ' '/VLLM::Worker/{print $2}'; }
step boot-log;  docker logs qwen38-flash-next 2>&1 | grep -E "Model loading took|Graph capturing finished|post-capture warm-up|GPU KV cache size" | sed 's/.*\] //' | cut -c1-120
step memramp;   python3 memramp.py "$N"
step exact;     python3 exact_boot.py save "$N-a0" | tail -1; python3 exact_boot.py save "$N-a1" | tail -1
step run_set;   ./run_set.sh "$N" quick > /dev/null 2>&1; tail -1 "results/$N/run.log"
step mem-after; echo "worker $(mib) MiB"
step done
