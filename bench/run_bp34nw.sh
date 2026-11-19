#!/usr/bin/env bash
# Record of the 2026-11-19 trial (bp34nw). It booted with files/mounts-bp34.txt = files/mounts.txt + the five
# files/v34 files then (incl. a qwen4_exp model.py copy and gpu_worker.py with the post-capture warm-up, #62560).
# Both were removed after the decision (bench/DECISIONS.md, 2026-11-19 09:40); to rerun, recreate that list.
# Attribution: bp34 mounts with the post-capture warm-up off; memory at boot, after run_set quick, after a second prefill pass.
cd "$(dirname "$0")"
mib() { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits | awk -F', ' '/VLLM::Worker/{print $2}'; }
MOUNTS=files/mounts-bp34.txt ./up.sh bp34nw || { echo "== bp34nw boot FAILED"; exit 1; }
docker logs qwen38-flash-next 2>&1 | grep -c "post-capture warm-up done" | sed 's/^/post-capture warm-up lines: /'
echo "== $(date -u +%T) boot worker $(mib) MiB"
./run_set.sh bp34nw quick > /dev/null 2>&1; tail -1 results/bp34nw/run.log
echo "== $(date -u +%T) after run_set worker $(mib) MiB"
python3 conc.py prefill --trial bp34nw-p2 > /dev/null 2>&1
echo "== $(date -u +%T) after a second prefill pass worker $(mib) MiB"
echo "== all done"
