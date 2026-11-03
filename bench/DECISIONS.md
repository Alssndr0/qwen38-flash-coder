# 4 concurrent requests, smooth: decisions

Goal (the user, 2026-11-02): serve v4gdnq as well as possible at 4 concurrent requests for agentic coding in
OpenCode (mostly) and Pi. Above all, no visible freeze in a decoding stream when another request arrives and
prefills. The newcomer may wait a little longer for its first token. Model changes are allowed if they help.

Method: the lab's protocol (~/projects/llm-tuning-lab/PROTOCOL.md) with quick tests while optimizing
(memory: optimize-fast-iteration). Tools in this folder: conc.py (scenarios), report.py (measures), mkrecipe.py /
up.sh (trial recipes in trials/), prof.sh (nsys), ../files/mbx_sched.py (step tracer + live pacing policy).

## Measures (fixed before any trial)

- stall scenario, per case (newcomer fresh 10k / 23k / 40k, delta 3k on 40k, delta 12k on 5k; 3 OpenCode streams
  decoding; 2 reps): keep % (decode speed of the 3 while the newcomer prefills, vs the 3 s before), freeze (longest
  gap between deliveries on any of the 3, ms), newcomer TTFT.
- agents scenario (4 workers, quick set: 8 recorded sessions, 58 turns, 1 s think time): stream_tps (decode tok/s
  per stream, pooled), total_tps, TTFT p50 / p90, gap p99 / max, gaps over 0.5 s and 1 s per 1000 tokens, stall %
  (share of decode time spent beyond 250 ms in a gap).
- capacity (c = 1..4 streams, nothing arriving): stream tok/s; prefill alone: prompt tok/s.

## Change classes and pass rules

- **S (scheduling, exact)**: outputs unchanged by construction. Pass: agents stall % and gaps-over-1 s improve, stall
  scenario freeze improves on every case, agents total_tps not more than 5 % below baseline, TTFT p90 not more than
  doubled. Among passing variants prefer the lowest freeze at the smallest TTFT cost.
- **T (kernels meant to be exact)**: as S, plus Q1 KL within the floor.
- **C (changes outputs: activation precision, checkpoint)**: Q1 NLL, Q6 code eval (EvalPlus), English check, tool
  validity; the user decides on a quality change.

## Log (newest first)

### 2026-11-03 17:05 UTC: trial `gdn8k` up for the user (expert kernel + MXFP8 GDN prefill, user: "put up the best
version with the FP8 variant so I can test it")
- bench/trials/gdn8k.yaml = recipe.yaml + MBX_GDN8 (live file cache/mbx-gdn8.json {"min_tokens": 256}; 100000000 =
  off) + smooth la 3.0 / lb 0.0062, refitted on this mix (bench/lbfit.py, 163 slices: 3.04 + 0.00615 * tokens, rms
  5.1 ms; 6,791 tokens 44.85 ms per layer vs 48.53 without GDN8). Model loading 69.87 GiB (+2.1 vs the default).
- Prefill alone 3.30 / 7.41 / 13.23 / 21.38 s (default with the kernel 3.51 / 7.88 / 13.91 / 22.50; this morning's
  default 3.67 / 8.26 / 14.58 / 23.61: -9.4 to -10.3 %). Stall, 1 rep (during the fit, lb 0.0062 / la 2.9): newcomer
  TTFT 6.66 / 2.57 / 5.46 / 11.33 s (default 7.23 / 2.69 / 5.41 / 11.77), worst freeze 364 / 244 / 429 / 412 ms.
- Not the default (class C, the user decides); ./run.sh boots the default again.

### 2026-11-03 16:15 UTC: slice cost model refit for the expert kernel (user: "retune the prefill-slicing cost setting")
- Fit (bench/lbfit.py on 181 "tlog" slices of a stall run, default boot with the kernel): per slice layer
  2.93 + 0.00662 * tokens ms, rms 5.5 ms; per bin measured / fit / old model (2.4 + 0.0071 n): 1,293 tokens 11.30 /
  11.49 / 11.58, 4,592: 33.64 / 33.33 / 35.00, 6,912: 48.53 / 48.69 / 51.48, 7,690: 53.99 / 53.84 / 57.00. The old
  model overestimated a full chunk's layer by 6 %, and the planner floors to whole layers, so slices often got one
  layer fewer than the gap target allows.
- A/B on one boot, live (setlive.py), stall bg 3 x 2 reps, old -> la 2.9 / lb 0.0066: newcomer TTFT delta12k@5k
  7.40 -> 7.23 s, delta3k@40k 2.67 -> 2.69, fresh10k 5.96 -> 5.41, fresh23k 12.22 -> 11.77; worst freeze 384 / 264 /
  344 / 357 -> 419 / 256 / 322 / 364 ms (single-rep runs of one config differ by 40-65 ms); slice steps 236 -> 209,
  their p50 184 -> 217 ms (target 220), p90 272 -> 276, p99 324 -> 307; keep 37-57 % -> 29-42 % (fuller slices).
- Agents quick x2 (vs the same kernel at the old model, moekc-agents / 2): stall 0.64 / 0.49 % (1.11 / 0.84), gap
  p99 258 / 264 ms (276 / 272), max 687 / 397 (706 / 514), no gap > 1 s, total 112.0 / 109.8 tok/s (104.9 / 106.1),
  TTFT p50 1.33 / 1.34 s (1.38 / 1.63).
- Taken: smooth: la 2.9, lb 0.0066 in recipe.yaml (default boot checked: the live file carries them). With MBX_GDN8
  the fit should be redone (README).

### 2026-11-03 15:45 UTC: expert kernel, outcome: the kit default (MBX_MOEK in recipe.yaml)
- Scale format v2 (files/mbx_moek.py pack_scales): one base per 8 positions x 8 k-groups (a producer lane's 64
  scales) at the start of the 16-value window that holds most of them, 4-bit offsets, the 0.24 % of scales outside
  in per-(expert, item, run) exception lists that the producer prefetches when it issues the run (per-run lists
  scanned at decode time cost 1-3 % at decode sizes). Lossless; 3.93 GB for the model (v1: 4.81). Model loading
  67.79 GiB (csf Marlin 68.21). Weights tiled in place, scales packed in expert chunks (less fragmentation at load).
- The experts take vLLM's own plain-scale preparation (the module's source without the image's csf patch), so the
  dense NVFP4 layers keep csf. MBX_MARLIN_CSF stays 1.
- Bench (layer 10, CUDA graphs, L2 flushed): decode T=4 / 8 / 16 / 32 / 64 / 128 vs csf Marlin 0.0 / +0.9 / -0.1 /
  -1.6 / -2.1 / -2.8 %; T=1-2 +11-16 % (rare with 7-token drafts); prefill layer 9.59 vs 12.23 ms. Correctness sweep
  and determinism as before; the serve's own code path checked at T=1..6912 (bench/moek/serve_path.py).
- Serve (boot moekc = recipe + MBX_MOEK, KV 22 GB, vs basecap / basedef / def48 on the csf Marlin recipe):
  exactness (bench/exact_boot.py, greedy decode, 6 agent prompts): KL per step before the first divergence vs the
  baseline 1.08-2.43e-2, the baseline's own A/A pairs 1.12-3.23e-2, divergence points alike: run-to-run level.
  Prefill alone 3.51 / 7.88 / 13.91 / 22.50 s (def48 3.67 / 8.26 / 14.58 / 23.61: -4.4 to -4.7 %). Decode step p50
  x1-x4 (two runs): 42.1-43.1 / 56.2-58.0 / 70.3-71.0 / 80.1-83.2 ms (basecap 42.6 / 57.1 / 71.2 / 85.1).
  Stall (bg 3, vs basedef): newcomer TTFT 7.25 / 2.62 / 6.02 / 12.30 s (7.52 / 2.82 / 6.08 / 12.59), decoders' worst
  freeze 380 / 262 / 375 / 359 ms (399 / 301 / 360 / 379), keep 39-50 % (33-48 %). Agents quick, two runs: stall
  1.11 / 0.84 % (def48 1.03 / 0.70), max gap 706 / 514 ms (693 / 407), no gap > 1 s, TTFT p90 3.07 / 4.37 s (3.62 /
  5.26); the def48 pair itself differs by as much. robust.py: abort, 66k, burst, logprobs all PASS.
- Taken as the default: MBX_MOEK "1" in recipe.yaml (switch "0" = the image's Marlin with tile 48). Also
  gpu-memory-utilization 0.88 -> 0.86: with the KV pinned it is only vLLM's startup free-memory check, which failed
  twice today when the page cache held the model files.
- Follow-ups: the slicing cost model (smooth: lb, 0.0071 ms per token and layer) predates the faster experts; a lower
  lb would give each prefill slice more layers at the same gap target (newcomers faster, decoders unchanged). T=1-2
  batches are slower than csf Marlin (split units too short); not seen with 7-token drafts.

### 2026-11-03 13:20 UTC: expert GEMM kernel, decode parity and the serve integration (boot `moek`)
- Compressed scales of its own (bench/moek/kcuda.pack_scales, files/mbx_moek.py): per run of 8 k-groups and 128
  columns a base + 4-bit offsets, exceptions (0.26-0.41 % of runs) in a side list; lossless (round trip checked),
  0.63 of the plain bytes (the image's csf: ~0.58). The producer warp copies one run pair per 2 stages and decodes it
  into plain scales in shared memory; consumers unchanged. Packed == plain bitwise.
- Persistent kernel (one CTA per SM, producer warp streams the next item's weights during the current one), epilogue
  from registers (gate_up: each warp holds 32 gate + the matching 32 up columns). Schedules: whole items handed out
  in order by an atomic counter (prefill; contiguous stream-k ranges were tried: balanced but 40 % slower at prefill,
  the SMs no longer share activation rows in L2, and +5-9 % at decode, 48 experts read at once); batches <= 12 tokens
  split gate_up along k (5 units, partials added by a second small kernel in a fixed order).
- Tiled weight layout (same bytes, reordered at load): one contiguous 8 KB block per (expert, 256 columns, 64-k
  stage) and one contiguous scale block per run pair, so a stage is a single bulk copy; this was worth 5-10 % at
  decode sizes (DRAM efficiency of many small scattered copies). Bitwise equal to the Marlin-layout path.
- Decode sizes vs the serve's csf Marlin, CUDA graphs, L2 flushed before each layer (bench/moek/test_tiled*.py):
  T=4 +1.2 %, T=8 +0.5 %, T=16 +0.7 %, T=32 -0.5 %, T=64 -1.4 %, T=128 -1.5 %; T=1-2 +9-20 % (tiny absolute, rare
  with 7-token drafts). Prefill layer (T=6912): 9.90 ms vs 12.16 (csf Marlin), silu fused; the top-k sum stays vLLM's.
- Correctness (bench/moek/correct_sweep.py, layer 10, T=1..10368, skews 0.6-1.5): error vs the bf16 reference equal
  to or below Marlin's at every size; deterministic run to run (determinism.py).
- Serve integration: files/mbx_moek.py (+ files/mbx_moek.cu, JIT build into cache/moek-build), MBX_MOEK=1 with
  MBX_MARLIN_CSF=0, installed from files/base_loader.py before the weights are processed (wraps vLLM's NVFP4 Marlin
  MoE preparation and fused_marlin_moe; other Marlin users unchanged).
- Exactness method across boots (bench/exact_boot.py): prompt logprobs are noisy on this serve (two identical runs on
  one boot: top-1 agreement 80 %), so the gate is greedy decode: per-step KL before the first divergence and the
  divergence points, kernel boot vs baseline against the baseline's own A/A pair (base-a0/a1, base-bg2 saved on the
  dev boot = recipe with KV 12 GB).

### 2026-11-03 12:10 UTC: expert GEMM kernel (user: "try to implement that custom kernel"), first results
- Real layer weights now in the harness (bench/moek/problem.py real_weights: layer 10 from the checkpoint).
- Where the time really goes (bench/moek/diag.py, opsplit.py, layer 10, T=6912): the serve's compressed-scale
  Marlin (csf) takes 10.99 ms for the two GEMMs; vLLM's plain-scale Marlin 9.05 ms on the same weights; plus silu 0.54
  and sum 0.9-1.05. csf is 2-4 % faster at decode sizes (fewer scale bytes) but ~20 % slower at prefill (it decodes
  scales per 48-row block). Plain Marlin already streams near the floor at 16-48 rows per expert.
- New CUDA kernel bench/moek/csrc/moe_w4a16.cu (reads Marlin's weight + plain scale layout, same arithmetic): one CTA =
  up to 96 rows of one expert x 256 columns; the row-fragment count is a compile-time variant (no MMA on padding);
  warps = 4 column slabs x 2 row groups (8 MMAs per activation fragment); weights by bulk copy (TMA) from a dedicated
  producer warp with full/empty mbarriers (a TMA issue blocks its thread while the copy queue is full: issuing from a
  math warp stalled the whole CTA, measured with in-kernel clock stamps); gate_up epilogue fuses silu_and_mul.
- Correctness: gate_up+silu bit-identical to Marlin + silu_and_mul on 99.97 % of outputs; layer vs bf16 reference
  0.0051 (Marlin 0.0057).
- Speed (layer 10, lognormal routing, T=6912): gate_up+silu 5.36 ms (Marlin plain 6.05, csf ~7.6); down 3.48;
  whole layer with the sum 9.85 ms (plain Marlin path 10.21, csf path 12.45). Decode sizes: equal to plain Marlin from
  T=8 up (all at the weight floor), slower at T=1-4 (0.105 vs 0.075 ms).
- Activation gathers cost ~0.5 ms of gate_up (x = 35 MB does not fit the 24 MB L2, each token row is re-read once
  per expert); the realistic gate_up floor incl. those reads is ~5.1 ms.
- Next: a compressed scale format the kernel reads (4-bit offsets + base along k, exceptions decoded by the producer
  warp), so the kernel can replace Marlin for decode too at csf's memory and byte count; then the small-batch path,
  serve integration and the gates.

### 2026-11-03 10:40 UTC: a faster expert GEMM kernel? (user: "can we not work on that?"), feasibility
- Ceiling (bench/moek/peak.py, mma.sync): bf16 106 TFLOP/s, fp8 213. One layer at a 6,912-token chunk: weight read
  6.2 ms, bf16 math 3.2 ms; Marlin (tile 48) ~12 ms for both GEMMs.
- Harness bench/moek/problem.py: one layer, random NVFP4 weights in the checkpoint layout, routing with lognormal
  expert popularity, bf16 reference; Marlin (the serve's call) matches the reference to 0.6 % (bf16 rounding), so
  the layout reading is right. Marlin there: 14.0-14.4 ms per layer with tile 48 (gate_up alone 8.09 ms).
- Triton prototypes (bench/moek/moek.py; same math as Marlin, outputs match the reference to 0.55 %):
  v1 one 64-row block per program: gate_up 9.47 ms, down 4.89, whole layer 15.6 ms (Marlin 14.4);
  v2 row groups (weight tile decoded once for up to 128 rows): 17.5 ms (the conditional dots lose pipelining).
  Ablation (try3.py, gate_up): no weight decode at all 8.97 ms vs 9.21 with it: the 4-bit decode is not the cost.
- Diagnosis: both kernels run at ~37-41 TFLOP/s on the padded rows. Marlin runs one CTA per SM (99 KB shared
  memory, 8 warps), so its weight stream (4.1 ms for gate_up) and its math (~3-4 ms) mostly take turns: ~8 ms.
  What would beat it: a CUDA kernel with a deep asynchronous weight pipeline (warps that only stream weights, warps
  that only multiply), reading Marlin's own layout so decode keeps Marlin. Realistic ~3-4 ms per layer, i.e.
  ~6-8 % prefill; a days-long kernel project, and the serve's compressed-scale (csf) format is closed (.so).
  Reported to the user before starting it.

### 2026-11-03 09:30 UTC: expert layers, outcome: Marlin row tile 48 is the kit default (MBX_MOE_BM in recipe.yaml)
- Default boot (./run.sh with MBX_MOE_BM "48"): prefill alone 3.67 / 8.26 / 14.58 / 23.61 s (fresh 10k-66k; before
  3.73 / 8.40 / 14.95 / 24.17). Agents quick, cold first run after boot: stall 1.03 %, max gap 693 ms, no gap > 1 s
  (the previous default's cold run final-runsh2: 0.77 %, 746 ms); warm second run: stall 0.70 % (g8a-off 0.75),
  max gap 407 ms, no gap > 0.5 s, stream 42.9 tok/s (40.5), TTFT p90 5.26 s (5.52), tok/delivery 3.86 (3.63).
- Bigger prefill chunks, follow-up: with a decoding batch, chunks stayed 6,912 tokens even at max-num-batched-tokens
  10368 and 10496 (only a prompt's final chunk grows); the scheduler's budget left after the decode and lookahead
  tokens falls just short of three 3,456-token blocks. Under load the gain was therefore only the bigger final chunk
  (newcomer fresh10k -13 %, fresh23k -4 %) and so was the cost (decoders' worst freeze +60 to +120 ms). Not taken.
- Why a prompt's last step freezes the decoders for ~350-470 ms (vs ~280 for other slice steps), measured with the
  new slice log ("tlog": true, GPU and CPU ms per slice): the final slice itself is short (fresh10k: 1 layer, GPU
  116 ms, CPU 9 ms), so reserving room for it in the planner (tried as "lf", reverted) cannot help; the extra
  ~150-250 ms is outside the slice's own work (first-token sampling / drafter / output merge path). Left as a lead.

### 2026-11-03 08:40 UTC: Marlin MoE row tile (files/mbx_moe.py, live) and bigger prefill chunks
- Row tile for prefill-size batches (>= 256 tokens), live on boot moelive (bench/moe_sweep.sh, prefill alone, fresh
  10k / 23k / 40k / 66k): vLLM's choice (64) 3.72 / 8.43 / 14.94 / 24.16 s; 16: 3.76 / 8.45 / 14.98 / 24.25; 32:
  3.70 / 8.35 / 14.80 / 23.97; 48: 3.67 / 8.21 / 14.54 / 23.58; 64: 3.80 / 8.47 / 14.92 / 24.21. Repeat 48 vs 0:
  3.67 / 8.21 / 14.53 / 23.55 vs 3.73 / 8.40 / 14.95 / 24.17. Tile 48: prefill -1.6 to -2.8 %.
- Exactness of tile 48 (exact2.py --warm-bg, 2 decoders in the background, results/m14-kl.json): next-token KL at
  or below the A/A pair in every case (fresh10k 0.012 / 0.157 vs A/A 0.028 / 0.153; delta12k@5k 0.104 vs 0.171;
  delta2k@66k 0.015 / 0.031 vs 0.056 / 0.008; delta3k@40k 0 = 0), top-1 the same. Exact at the run-to-run level.
- max-num-batched-tokens 13824 (4 blocks, boot moe14k) + tile 48, prefill alone: 3.37 / 7.94 / 14.33 / 23.30 s
  (-9.4 / -5.8 / -4.1 / -3.6 % vs default). Stall scenario (3 decoders, vs g8s-off): newcomer fresh10k 6.17 -> 5.38 s,
  fresh23k 12.79 -> 12.25, fresh40k 22.83 -> 21.79, delta12k@5k 7.56 -> 7.69, delta3k@40k 2.73 -> 2.72; but worst
  freeze of the decoders up: fresh23k 362 -> 487 ms, fresh40k 456 -> 482 (the last step of a chunk is not sliced
  and grows with the chunk). Agents quick: first run after boot had one 2.35 s gap (first-use; the API server had
  ~0.9 GB swapped out), the second run none (max 404 ms, stall 0.34 %, total 118 tok/s). Not taken as is: the
  freeze is the user's first criterion. 10368 (3 blocks) tried next as the middle.

### 2026-11-03 07:40 UTC: expert layers (user: "work on the expert layers"), profile and Humming
- Where the time goes (bench/prof/tr-base-81.1, the 6,912-token chunk of a fresh 10k prompt, per layer): 49.5 ms in
  all; routed-expert GEMMs (Marlin csf, row tile 64) 12.9 ms (gate_up 8.2 + down 4.8), i.e. ~25 TFLOP/s and 2.1x the
  expert weight-read time (1.42 GB per layer = 6.1 ms); moe_sum 0.9, activation 0.6, routing 0.1, shared expert
  (MXFP8) ~1.4. Dense Marlin (GDN projections, PLE / HC) is another 12.9 ms.
- Humming MoE backend (`moe-backend: humming`, W4A16, same 4-bit weights, boot hum16): prefill alone -5 to -6 %
  (warm rep: 3.56 / 7.95 / 13.99 / 22.74 s vs 3.72 / 8.41 / 14.91 / 24.13), but decode slower at every concurrency
  (capacity c=1 138.7 vs 148.8 tok/s, c=4 238 vs 255 total). Decode dominates agent time: not adopted. A prefill-only
  Humming path would need a second copy of the 68 GB expert weights.

### 2026-11-03 04:25 UTC: C1 results (boot gdn8, every comparison paired live on the same boot)
- Q1 (klprobe, KL corpus): NLL 1.0247 with MXFP8 prefill (gate <= 1.0335: pass); the same boot with it off 1.0164,
  the earlier v4gdnq capture 1.0215, f5c 1.0254. KL on vs off 0.075 (off vs the earlier v4gdnq capture 0.041),
  on vs f5c 0.059: prefill now scores like f5c (MXFP8 GDN), a little above the 4-bit GPTQ projections.
- Q6 (EvalPlus): 453 / 542 = v4gdnq 453 (only v4gdnq 4, only gdn8 4, McNemar p 1.0): pass.
- Q2 / Q3 (quick replay, 28 requests): CJK replies 0 / 0, tool args valid 35/35 and 37/37: pass.
- Teacher-forced next-token KL on OpenCode prompts with 2 decoders (exact2.py --warm-bg): on vs off within the A/A
  spread on all 4 cases, same top-1 agreement.
- Speed, prefill alone: -4.7 % (10k) to -5.7 % (23k-66k). Stall scenario (3 decoders, slicing on, lb 0.0067 for
  on): newcomer TTFT fresh10k 6.17 -> 5.48 s, 23k 12.79 -> 11.93, 40k 22.83 -> 21.28, delta3k@40k 2.73 -> 2.64,
  delta12k@5k 7.56 -> 7.20; freeze unchanged (268-456 vs 270-434 ms). Agents quick: stall 0.75 -> 0.74 %, gaps > 0.5 s
  0 / 0, gap max 388 -> 365 ms, stream 40.5 -> 42.9 tok/s, TTFT p50 1.34 -> 1.35, p90 5.52 -> 4.45 s,
  tok/delivery 3.63 -> 3.76.
- Watch item resolved: the quick replay showed g8 -13 % tok/s (tokens per step 3.78 -> 3.29, bootstrap CI over 5
  sessions [-26, -6]). Not the change: the replay shares the prefix cache between its arms, and in 3 of the 5
  sessions the g8 arm reused w4's cache and prefilled only 200-580 tokens itself in turns under 256 tokens, i.e. it
  never ran MXFP8; those 3 show the biggest deficits (-29, -20, -6 %). Session-level noise from diverging samples is
  ~30 %, so 5 sessions cannot resolve this. Direct check (exact.py, greedy, fresh salt per run, single stream, 7
  prompts x 400 tokens, two runs per arm): tokens per delivery 4.26 (4-bit) vs 4.18 (MXFP8), signs mixed (3 up, 4
  down), within the A/A spread; TTFT -4 to -6 % on every prompt. Decode itself is the 4-bit path by construction.
- Verdict under the pre-registered rules: every quality gate passes, TTFT improves with stalls and gaps unchanged.
  Cost: +2.14 GB, prefill NLL at f5c's level (+0.008 vs the 4-bit projections). The user decides; the boot stays up
  with the live switch off (= v4gdnq exactly).

### 2026-11-03 03:20 UTC: pre-registration C1, prefill GDN projections on the MXFP8 originals (class C), boot `gdn8`
- Asked by the user ("try this", 2026-11-03). Code: files/mbx_gdn8.py (hook in files/base_loader.py, MBX_GDN8=<AZ dir>):
  in_proj_qkvz and out_proj of the 36 GDN layers get a second copy from the AZ checkpoint's MXFP8 weights (+2.14 GB);
  one opaque op per call runs batches of >= MBX_GDN8_MIN_TOKENS (256) tokens on MXFP8 W8A8 (b12x; out_proj in
  2048-row pieces above 3456 rows), smaller ones on the 4-bit Marlin path as before. Decode (<= 128 rows, CUDA
  graphs) is unchanged by construction. Live switch cache/mbx-gdn8.json {"min_tokens": N}: a huge N = v4gdnq exactly.
- Kernel bench (bench/gdn8_bench.py, before the boot): qkvz MXFP8 1.27x at 256 rows, 1.8x from 512 (6,912: 7.7 ->
  4.4 ms); out_proj 1.5x at 256, 1.1-1.35x above (whole 6,912 rows is 0.56x: hence the pieces).
- Measured before this entry (speed only): prefill alone, paired live (off / on, 2 reps each, stable to 0.03 s):
  fresh10k 3.73 -> 3.56 s (-4.7 %), 23k 8.42 -> 7.95 (-5.6 %), 40k 14.90 -> 14.06 (-5.7 %), 66k 24.13 -> 22.77
  (-5.7 %). The 10 % estimate was too high: the projections are ~24 % of a GDN layer, MXFP8 halves only qkvz.
- Quality gates (the v4gdnq quick gate, vs the FP4-Local reference as there): Q6 plus passes >= 448 (reference mean
  453 - 5) and McNemar vs codeeval-v4gdnq not significant for a loss; Q1 NLL on the KL corpus <= 1.0335 (reference
  1.0275 + 0.006); Q3 tool args valid on the quick replay; English replies with CJK 0. Also reported: teacher-forced
  KL of on vs off on OpenCode prompts (exact2.py --warm-bg), against the A/A floor.
- Speed (paired live on the same boot): stall scenario and agents quick, on vs off; pass if newcomer TTFT improves
  with stall % and gaps not worse. The user decides on adoption.

### 2026-11-03 00:20 UTC: default boot (./run.sh, recipe.yaml smooth: section) validated, b12x tried
- Agents on the default boot `final-runsh2` (with the new boot warm-up of the sliced path, which removes first-use
  JIT stalls; `final-runsh` without it had one 2.1 s gap): stall 0.77 %, gaps > 1 s 0, gap p99 249 / max 746 ms,
  stream 41.3 tok/s, TTFT p50 1.71 / p90 6.92 s, total 106.7 tok/s (base 114.6), tok/delivery 3.67.
- b12x MoE backend (prefill speed): W4A16 (VLLM_B12X_MOE_FP4_FORCE_A16) prefill 2.39-2.71k tok/s = Marlin, capacity
  the same; W4A4 2.48-2.89k tok/s (+7 % at 66k), capacity the same or lower; both not adopted (W4A4 would also
  change outputs). Marlin stays.
- Left for the user (class C): GDN in/out projections from the MXFP8 originals for prefill-size batches (+2.1 GB,
  est. ~10-12 % faster prefill, prefill numerics change).

### 2026-11-02 23:35 UTC: S2 layer-sliced prefill, results (boot tr-slice)
- Four exactness bugs found by the teacher-forced tests and fixed (each re-tested): (1) admission with the whole
  prompt allocated broke the mamba-align bookkeeping -> intercept each chunk exactly as the scheduler planned it;
  (2) other prefills rode in decode steps -> hold all prefills while a job runs; (3) the final chunk stopped one
  token short broke prefix-cache state checkpoints (later turns on a sliced turn read wrong state) -> the last chunk
  runs whole and its last slice is a normal sampling step, the first token merged into that step's output;
  (4) synthetic steps fed mbx_depth, which disabled drafting for a reused slot -> hidden from it.
- Exactness (bench/exact2.py, exact3.py): next-token KL at the A/A floor on fresh, delta and multi-chunk prompts, and
  on later turns that hit a sliced turn's prefix cache (delta3k@40k 0.0000, delta12k@5k 0.0011-0.024, delta2k@66k
  0.0069, pi2to3 0.0000; A/A 0.0000-0.0136). Greedy runs diverge no earlier than A/A (Marlin atomic adds make the
  serve nondeterministic).
- Robustness (bench/robust.py): abort mid-prefill, 66k prompt, 8-request burst, logprobs request: all PASS.
- Agents quick, best point `ag-gap220` (gap target 220 ms, busy 260, late 320 after 4 s, idle 600):
  stall 0.75 % (base 12.7), gaps > 0.5 s 0.14 / ktok (3.47), > 1 s 0 (1.05), gap p99 258 ms (588), max 696 (3044),
  stream 44.1 tok/s (38.9), tok/delivery 3.75 (3.58), TTFT p50 1.49 s (0.88): < 1k new 0.97 (0.66), 1-3k 1.67
  (1.11), 3-10k 3.66 (3.56), > 10k 9.56 (8.57); total 105.1 tok/s (114.6).
- Verdict under the S rule: stall and gaps pass by far; total -8 % (rule: within 5 %) and TTFT p90 6.1 s vs 3.39
  (rule: at most doubled: pass). Total throughput misses the 5 % bound: reported to the user as the price of
  smoothness; slicing stays on live (switch: cache/mbx-smooth.json "mode": "off").

### 2026-11-02 19:10 UTC: pre-registration S2, layer-sliced prefill (class S, must be exact), boot `tr-slice`
- Why: the profile (bench/prof/tr-base-81.*) shows a prefill-carrying step streams most expert weights (MoE ~4 ms per
  layer at 536 tokens, ~190 ms per step), so any prefill step costs >= ~300 ms with 3 decoders. Pacing cannot go
  below that. Slicing runs the newcomer's chunk a few layers per step after the normal decode step.
- Code: files/mbx_sidecar.py (worker), files/v54/model.py `_forward_slice`, files/mbx_sched.py mode "slice".
- Gate 1 (exactness, bench/exact.py): greedy probe after a sliced vs unsliced prefill, 3 cases, 2 decoders in the
  background; A/A pair (off vs off) as floor. Pass: tokens per delivery after the prefill within 10 % of off (the
  drafter context is right), logprob differences at the A/A level, no crash, no errors.
- Gate 2 (S rule): stall freeze below the cap-512 point (375 ms) on every case, agents stall % below 8.5 (the best
  pacing point) with TTFT p90 better than cap 1024 share 0.67 (8.55 s).

### 2026-11-02 19:00 UTC: S1 on agents (live, tr-base)
- cap 2048: gap max 1601 ms (base 3044), gaps > 1 s 0.03 / ktok (1.05), stall 12.1 % (12.7), TTFT p90 2.33 s (3.39),
  total 104.4 tok/s (114.6), stream 41.7 (38.9).
- cap 1024 share 0.67: gap max 574, gaps > 1 s 0, stall 8.5 %, TTFT p50 1.64 / p90 8.55 / max 18.4 s, total 104.7,
  stream 43.2.
- Reading: pacing removes freezes > 1 s but the small prefill steps of every turn still stall ~300-500 ms; the
  smoother point costs a lot of TTFT. Kept as fallback; slicing (S2) next.

### 2026-11-02 18:40 UTC: S1 results (tr-base boot, live policies)
- Check 0 passed: tracer neutral (stall freeze 2530 vs 2548 ms fresh10k; capacity c=3/4 76.3/64.3 vs 77.9/63.6;
  c=1/2 tok/s vary +-25 % run to run with acceptance, so decode capacity is judged by step time from now on:
  x1 41, x2 58, x3 67, x4 84 ms).
- Step cost model (stable across all sweeps, rms 16 ms): ms = 130 + 0.341 * prefill_tokens + 17 * decoding_streams
  (context term ~0). So prefill compute is 2.93k tok/s and every prefill-carrying step pays ~110-130 ms more than a
  decode-only step, even at 256 tokens. Small chunks lose throughput to that fixed cost.
- Sweep A (gap 0), freeze ms / fresh10k TTFT s / keep %: cap 512: 375 / 8.0 / 22-43; 1024: 540 / 5.5 / 15-26;
  2048: 900 / 4.8 / 10-29; 3456: 1360 / 4.2 / 8-27 (baseline 2550 / 4.1 / 4-14).
- Sweep B (time share): cap 1024 share 0.5: freeze 540, keep 56-65 %, fresh10k TTFT 10.5 s, delta3k@40k 4.0 s;
  cap 512 share 0.5: 380 / 68-80 % / 14.2 s; cap 1024 share 0.67: 535 / 50-58 % / 8.0 s. Share accounting checked
  in the trace (prefill 5.3 s vs decode-only 5.2 s at 0.5).
- Reading: pacing alone can bound the freeze, but every point costs the newcomer 1.3-3.5x TTFT because prefill is
  slow (2.9k tok/s) and each prefill step has a ~120 ms fixed cost. Next: profile what that fixed cost and the
  0.34 ms/token are, then make prefill faster.

### 2026-11-02 18:15 UTC: pre-registration S1, prefill pacing (class S), boot `tr-base`
- Serve: recipe.yaml + `--scheduler-cls mbx_sched.SmoothScheduler` + tracer. Policy live in cache/mbx-smooth.json.
- Check 0 (neutrality, mode off): capacity c=1..4 within 5 % of base-notrace, stall freeze within 10 %.
- Sweep A, gap 0, cap 512 / 1024 / 2048 / 3456: stall cases fresh10k, fresh23k, delta3k@40k, delta12k@5k, 1 rep.
  Output: freeze / keep / TTFT per cap, and T(chunk) from the trace (cost model in steps.py).
- Sweep B, time share: cap from A at share 0.3 / 0.5 / 0.7.
- Candidates (best two) then run agents quick; rule as in "Change classes" (S).

### 2026-11-02 18:07 UTC: baseline (current recipe.yaml, no tracer) `base-notrace`
- prefill alone ~2,600-2,700 tok/s at 10k-66k (linear: per-token compute, not attention).
- capacity (bg prompts draft unusually well, relative only): 148.8 / 89.4 / 77.9 / 63.6 tok/s per stream at c=1..4.
- stall (delta cases rep 0 only: rep 1 reused rep 0's cached newcomer, fixed in conc.py after this run):
  fresh10k TTFT 4.08 s keep 11 % freeze 2548 ms; fresh23k 8.96 / 7 / 2576; fresh40k 15.64 / 4 / 2622;
  delta3k@40k 1.92 / 17 / 848; delta12k@5k 5.16 / 7 / 2577. Each 6,912-token chunk is one ~2.5-3.2 s step.
- agents quick (62 turns, 259 s): stream 38.93 tok/s, total 114.6, TTFT p50 0.88 / p90 3.39 / max 9.26 s,
  gaps p50 78 / p99 588 / max 3044 ms, 1.05 gaps > 1 s per 1000 tokens, stall 12.7 %.
