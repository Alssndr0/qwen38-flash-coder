# bench: concurrency and smoothness

Tools behind the "Smooth with several agents at once" section of the kit's README. Stdlib Python only; run from
this folder against the serve on :8000. Decisions and results, newest first: [DECISIONS.md](DECISIONS.md).

| Tool | What it does |
|---|---|
| `conc.py stall` | 3 OpenCode streams decode, a newcomer arrives (fresh 10k / 23k / 40k, or a delta on a cached session): freeze and slowdown of the 3, newcomer TTFT |
| `conc.py agents` | 4 workers replay 8 recorded OpenCode / Pi sessions (62 turns) turn by turn: speed, TTFT, gaps, stall % |
| `conc.py capacity` / `prefill` | 1 to 4 streams with nothing arriving; one fresh prompt alone |
| `report.py T [T2 ...]` | measures per scenario, trials side by side (`--steps` adds step times from the trace) |
| `steps.py T` | step trace analysis: step time by composition, fitted cost model |
| `exact2.py` | exactness, teacher-forced: next-token top-20 distribution of the same tokens under each policy (`--warm-bg`: the earlier turn is prefilled under the policy too, so a later turn tests its prefix cache) |
| `exact3.py` | exactness for unsliceable-by-logprobs paths: greedy sequences (`--bg 0` + `"force": 1` for a deterministic run) |
| `robust.py` | abort mid-prefill, 66k prompt, 8-request burst, logprobs request |
| `sweep_cap.py`, `run_agents_policy.sh` | run scenarios under live policies (cache/mbx-smooth.json), no restart |
| `mkrecipe.py`, `up.sh` | trial recipes in `trials/` (recipe.yaml with keys changed), booted through run.sh |
| `prof.sh`, `prof.py` | boot a trial under nsys and profile a prefill, a mixed step, 4-stream decode |
| `moe_sweep.sh` | Marlin MoE row tile on prefill batches, live (`MBX_MOE_LIVE`): prefill alone per tile, side by side |
| `exact_boot.py save NAME` / `compare A B` | exactness across boots (greedy decode KL and divergence vs the baseline's A/A pair) for changes with no live switch |
| `moek/` | the expert-kernel harness: one real layer (problem.py), the kernel's Python side (kcuda.py, builds files/mbx_moek.cu), correctness / determinism / speed scripts; `moek/run.sh X.py` runs one in a side container (`CSF=0` plain Marlin scales, `NCU=...` under Nsight Compute) |
| `lbfit.py LOG` | fit the slice planner's per-layer cost (`la`, `lb` in `smooth:`) from the worker's slice log (`"tlog": true` in the live policy) |
| `setlive.py key=value ...` | change keys of the live slicing policy (`cache/mbx-smooth.json`); `key=-` removes one |

The step tracer (`"trace": true` in the policy, or env MBX_STEP_TRACE) writes one JSON line per engine step to
`../cache/steptrace/`: when it was scheduled and finished, how many streams decoded, which prefill chunks it carried.
Results land in `results/<trial>/`.
