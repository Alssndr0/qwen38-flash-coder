#!/usr/bin/env bash
# Stall breakdown, pipe use and L2 traffic of one kernel launch: ncu_stalls.sh REGEX [env...] (runs prof.py)
re="$1"; shift
S=""; for s in barrier long_scoreboard short_scoreboard wait math_pipe_throttle mio_throttle not_selected selected dispatch_stall lg_throttle no_instruction branch_resolving misc; do S="$S,smsp__warp_issue_stalled_${s}_per_warp_active.pct"; done
M="gpu__time_duration.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,smsp__issue_active.avg.pct_of_peak_sustained_active,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,launch__registers_per_thread$S"
env "$@" NCU="--kernel-name regex:$re --launch-skip 2 --launch-count 1 --metrics $M" "$(dirname "$0")/run.sh" prof.py 2>&1 | grep -E "^\s+(smsp|sm__|lts|gpu__|l1tex|launch)" | sed 's/smsp__warp_issue_stalled_/stall_/; s/_per_warp_active.pct//' | awk '{printf "  %-58s %12s %s\n",$1,$3,$2}'
