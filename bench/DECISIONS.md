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

### 2026-11-19 09:40 UTC: 0.34 backport, outcome: #66048 + #65619 + #64675 adopted (kit default, live on :8000); #62560 dropped
- Exactness (exact_boot greedy, 6 agent prompts; prompt-position KL is ~1.15 within every arm too, so only the greedy part is
  read): KL per step before the first divergence, base34 A/A 2.12e-2, bp34 A/A 7.81e-2, across 2.05e-2 / 5.39e-2; adopted boot
  vs base34 2.36e-2 / 2.42e-2, divergence points [1, 58, 50, 7, 2, 85] vs base A/A [1, 58, 36, 3, 1, 85]. Run-to-run level.
- Speed, run_set quick, base34 -> bp34: agents total_tps 107.45 -> 108.88, stream_tps 40.68 -> 41.49, TTFT p90 4.66 -> 3.38 s,
  stall 0.94 -> 0.55 %, max gap 792 -> 364 ms; prefill alone within 0.5 %; stall scenario equal. Capacity per-stream tok/s
  +24 / -15 / -4.6 / -0.8 % at c=1..4 is acceptance: tokens per delivery 4.92 -> 6.35 and 5.88 -> 4.84 at c=1 / 2. Step time
  (delivery gap p50) 41.6 / 59.0 / 70.8 / 84.7 -> 43.2 / 57.0 / 69.9 / 83.3 ms; at equal tokens per delivery c=1 is -0.4 to
  +0.8 ms and c=2 -1.4 to -2.9 ms (no patch can do either: cross-boot noise). The c=2 capacity line misses the 5 % rule by the
  letter; read as no loss (acceptance), like bp33's total_tps.
- Memory (nvidia-smi, VLLM::Worker): boot base34 96,161 MiB, bp34 95,437, bp34 without the post-capture warm-up 95,435, adopted
  95,435 (#65619: -726 MiB, 11 x 67 MB expected). The decode ramp (1..16 streams x 2) grows nothing in any arm, so the sampler
  growth #62560 fixes upstream does not happen here. run_set quick grows the worker: base34 +5,096, bp34nw +3,560, bp34 +1,868
  MiB; a second prefill pass adds 0 (a plateau from large fresh prefills, not a leak). base34 vs bp34nw differ by 1.5 GB with no
  relevant change, so the warm-up's 1.7 GB is within single-run spread. #62560 dropped by the rule (it cuts no growth it
  targets); a real ~1.5-3 GB working-set effect would need repeats. Note for the OOM history: a fresh boot sits ~5 GB below its
  working set after long fresh prompts; the user's own 27 h old serve was still at the boot value (cache-hit traffic).
- Parser in the serve: a streamed `write` tool call parses (filePath + 9-line content, 11 argument deltas). Typical calls
  (348 recorded, delivered 10-20 chars at a time): parser CPU per whole call median 78 -> 48 us, p90 1.0 -> 0.5 ms, max
  27 -> 14 ms. Not user-visible (a call takes the model 1-3 s, on the GPU, in parallel); the gain is robustness against a
  looping call, not speed.
- Files: files/v34/ (rejection_sampler_utils.py = the 0.33 file + #66048, moved from files/v33/; qsa.py; parser/qwen3.py),
  files/v54/model.py (+ the shared buffer pool), mounts.txt. Tools: memramp.py, parser34_test.py, bp34_gates.sh, run_bp34*.sh
  (they used files/mounts-bp34.txt, removed with the dropped worker file). Committed and pushed 2026-11-19.

### 2026-11-19: pre-registration, backport from vLLM 0.34.0 (the user: "Yes proceed" to the four fixes below)
- Base `base34` = the kit default (./run.sh, KV 22G, no GDN8), freshly booted. Candidate `bp34` = the same recipe with
  `MOUNTS=files/mounts-bp34.txt` (run.sh now takes MOUNTS; default unchanged), five files in files/v34/:
  - #66048: `num_warps=1` on `_insert_resampled_kernel` (copy of files/v33 rejection_sampler_utils.py).
  - #62560, adapted: after capture_model, warmup_kernels runs once more (gpu_worker.py from the image). Upstream re-warms the
    plain sampler at max_num_seqs rows, which misses our verify path (16 x 8 fp32 logits rows, 127 MB, never chunked: the limit
    is 1,081 rows at vocab 248,320); warmup_kernels already runs 16 requests x 7 drafts at the warm-up sampling params, but
    before capture, whose empty_cache drops its blocks (boot log: capture "took -0.58 GiB"). MBX_POST_CAPTURE_WARMUP=0 = off.
  - #65619: the 12 QSA layers share one top-k buffer (8192 x 2052 int32, 67 MB each, ~0.74 GB saved); qsa.py from the image
    + files/v54/model.py. Safe here: each layer writes and reads its selection within its own call, no side streams, no MTP.
  - #64675, converter part (0.32 has no value extender): files/v34/vllm/parser/qwen3.py. Checked before boot
    (bench/parser34_test.py, CPU): 0 mismatches vs the image's converter over 113,113 streamed conversions (348 unique recorded
    tool calls + 8 edge cases, every prefix and 1-12 char chunks, one cache across several calls). Cost, streamed every 16
    chars: 400-param loop 395 -> 4 ms, 200 params 175 -> 7 ms, one 70K-char file write 17.2 -> 8.8 s (the open value is still
    re-scanned; upstream's value extender is the rest).
- Measures per arm, right after its boot, in this order (bench/bp34_gates.sh): memramp (worker memory from nvidia-smi at boot,
  then 1 / 2 / 4 / 8 / 16 streams x 2 ramps, real sampling, new tool bench/memramp.py), exact_boot x2, run_set quick, worker
  memory after. Smoke on the 27 h old live serve: 96,161 MiB, no growth over the ramp (already grown).
- Pass rule (as bp33): speed: capacity, agents total_tps and prefill not more than 5 % worse, TTFT p90 not worse by more than
  10 %. Exactness: greedy divergence and KL across arms at base34's own A/A floor. Memory: #65619 shows as lower memory at
  boot; #62560 must cut growth over ramp 1 (bp34 end vs base34 end, from boot); if base34 grows less than 0.2 GB over the ramp
  and run_set, #62560 has nothing to fix and is dropped. Fail on any one: drop the responsible patch, not the set. All exact or
  no output change, so no Q6 unless exactness fails.

### 2026-11-19: vLLM 0.34.0 review (we run 0.32.0 = myllmbox v5.4 + files/v33; myllmbox has no newer image, last tag v5.4 2026-10-27)
- No decode speed gain for this serve. Applies, exact or reliability, small mountable patches:
  - #66048: `_insert_resampled_kernel` reads `num_sampled`, stores +1, launches with 4 warps: cross-warp race (reported on 0.32.0,
    crash at an out-of-range index). Our files/v33 copy has it, and it runs on every sampled step (43 / 43 in prof/bp33-81.2). By
    the code, a late warp also stores the same token at num_sampled + 1 and the count + 2 (a repeated token) when it does not crash.
    Fix: num_warps=1. Rare; no cost.
  - #62560 (+ its prerequisite #57926, in 0.33.0, not in our backport): the worker frees the sampler's cached fp32 logits blocks at
    capture_model's empty_cache, so memory grows each time a larger batch first arrives (Qwen3-4B: +1.77 GB; peak -1.6 GB with the
    fix). Hypothesis to test, not established: the 2026-11-04 10:15 NVRM kill came at the start of the first 16-stream run.
  - #65619: one QSA top-k buffer for all 12 QSA layers instead of one each (8192 x 2048 int32 = 64 MiB per layer here): about
    0.7 GB more headroom (KV is pinned, so it does not grow the pool).
  - #64675: our `qwen3_xml` is Qwen3EngineToolParser with `_qwen3_arg_converter`; closed parameters were re-parsed every streamed
    step (a looping call saturated the API server and failed /health upstream). Not seen in our evals.
- Comes with an upgrade, not worth a backport alone: #58868 (prefault mmap pages on integrated GPUs: upstream Flash-Next NVMe load
  475 -> ~50 s; ours already loads in 124 s, boot ~4.5 min, gain unknown); #65466 / #65773 (we run the mp executor; moves admission
  work off the wait, a few ms of TTFT); #62702 / #65300 / #65831 (per-request CPU on 100k-token prompts, a few ms); #62748 (the main
  part is already in our runner).
- Does not apply: #59958 PLE disk offload (the image already reads the table from NVMe, async; upstream's costs -3.5 % decode,
  +11 % TTFT); #65874 fused GDN MTP kernel (our verify runs `_gdn_recoverssm_verify_kernel` / `_commit_gdn_state_kernel`);
  #64425 / #64999 / #64509 / #64723 / #51012 (Qwen3.5 model code); #66064 (GDN prefill gather/scatter, ~0.1 % of our expert-bound
  prefill); #65100 (resample kernel is 24 us per step here); #62956 (GLM-only SM121 GEMM plans); #65812 (checkpoint has no KV
  scales); #62318, #60490, #64820, #63761 (load layouts / >131k-token steps); #64059; #65033 (Mamba2); #63382 / #61755 (fused
  act+quant kernel not on our path); #65198 (drafter KV already fp8).
- Larger lead, not a backport: adaptive verification (per-request verify depth) is already in the 0.32 image but off here. 0.34
  fixes its mid-serving torch.compile stalls (#63421) and estimator precision (#65229); on GDN it needed graph changes (#60229, GLM)
  that our RecoverSSM path would also need. Upstream gain +6.5 / +12.7 % at depth 4 / 5, mostly at large batches.
- Side note from the profile: `_topk_topp_kernel` is 371 us per step at x1 (~0.9 %), outside this release.

### 2026-11-04 13:40 UTC: decode analysis (the user: decode must get faster)
- No regression from the backport: decode step p50 x1-x4 41.6-41.9 / 57.2-58.5 / 68.7-70.0 / 82.9-83.7 ms (unpatched moekc 42.1-43.1 / 56.2-58.0 /
  70.3-71.0 / 80.1-83.2). Agents total_tps moved with out_tokens; per-stream decode did not fall.
- Profile (nsys, bp33, prof/bp33-81.*), ms per step at 1 / 4 streams: routed experts 16.3 / 48.4, shared expert MXFP8 6.0 / 7.2, HC INT8 4.3 / 4.5, GDN
  qkvz and other Marlin 6.4 / 6.8, GDN out 1.6 / 1.9, GDN verify + commit 2.5 / 7.8, QSA 1.9 / 3.2, head INT8 1.7 / 1.7, MoE routing 2.1 / 1.5,
  router BF16 1.2. Dense layers cost the same at any M (weight reads, memory-bound); experts grow with tokens: ~1.9 ms per verified token at x1.
- Clock (step p50 at 2200 / 1800 MHz): x1 41.6 / 45.0, x2 57.9 / 60.7, x3 70.0 / 72.4, x4 83.7 / 83.5. Clock-bound share 37 / 22 / 15 / 0 %; projected at
  3003 MHz +11 / +6 / +4 / 0 %. Not tested above the user's 2200 cap. GPU ~50 C under this load.
- Ruled out: memory compaction before boot (x1 41.9); verify depth (with acceptance 0.80 .61 .48 .38 .31 .26 .21 and 1.9 ms per token, K=7 97.0 tok/s, K=6
  96.4, K=5 94.6); 4 experts (NLL +0.054); the public DSpark drafter (behind MTP on its own card, DFlash2 ahead of MTP on its).
- Ranked: (1) DFlash2 fine-tuned on agent sessions against our exact checkpoint (SpecForge model + our capture; exact; all concurrency; a DSpark run gained
  37 % from a low base; ours untested); (2) clock cap, the user's call; (3) shared expert and HC, own kernels or GPTQ 4-bit, est. 3-7 % at x1, 2-3 % at x4.

### 2026-11-04 12:40 UTC: backport adopted (`bp33`: gdn8k + the backport + KV 20G, live on :8000)
- Final run on KV 20G (677,205 tokens): no crash, no NVRM errors, min MemAvailable 4.81 GiB (3.75 at 22G). P3 0.27 s, 114 tokens. Agents
  stream_tps 43.46, total_tps 108.16 (out_tokens 26.8k), TTFT p90 4.27, stall 0.61 %, no gap over 1 s. Prefill sum 44.5 s (base 44.7).
- Quality: langcheck 0 CJK and 0 wrong language in 400; empty in thinking mode 18 / 160 (earlier configs 13-21, budget cuts). Replay 20
  requests: tool-valid 1.000, 88.79 tok/s, 3.710 tok/step. Q6 greedy over three patched runs 453 / 454 / 449 vs base 454 (worst pair: only
  base 7, only bp33 2, McNemar p 0.18; the kit's floor is 448). Q6 sampled 446 vs 446.
- Pass rule, one borderline: agents total_tps -5.0 / -1.4 / -5.3 % against a 5 % limit, with stream_tps -0.3 / +5.6 / +4.7 %: total_tps
  follows out_tokens here (fewer tokens, less overlap), so it is read as no loss. No speed gain is claimed for decode: tok/s differences are
  within the bench's run-to-run range. The measured gain is the next-turn TTFT.
- Open: the 10:15 crash stays unexplained (once in four runs, none on 20G). Whether the patches contributed was not tested.
- Defaults: recipe.yaml has the backport and stays at 22G without GDN8. `bench/up.sh bp33` boots the live flavor.

### 2026-11-04 11:40 UTC: backport results so far, and a 20G KV pool for this flavor
- calib control `nospec` (no drafting, 102.8k tokens): top-1 excess +0.34 points, z +4.5, same pattern as bp33-large (+0.43, z
  +5.9). So the excess is an offset of the test or of sampling in general, not block verification. Correction: an interim message
  to the user called base33's +3.4 proof of the bug on the real model; it is not. calib cannot see #61161 at this
  resolution (with vs without drafting differ by 0.09 +- ~0.13 points). The bug stands on the model-free script only.
- bp33 vs base33: P3 next turn TTFT 0.32 -> 0.27 / 0.28 s, recomputed 182 -> 118 / 114 tokens (the kept prefix block). Q6 greedy 454
  -> 453 / 454; Q6 sampled 446.0 -> 446.0 (CI -7.5 / +7.5). Agents stream_tps 41.51 -> 41.38 / 43.82, total_tps 114.2 -> 108.5 /
  112.7 (out_tokens 32.5k -> 24.9k / 27.7k, the replay's own range is 25-34k), TTFT p90 4.75 -> 4.63 / 4.55, tokens per delivery
  3.60 -> 3.73 / 3.85, no gap over 1 s. Prefill sum over the 4 sizes 44.7 s -> 44.9 / 45.6. Exactness at the A/A floor.
- One crash: serve killed at 10:15:37 (exit 137, NVRM out of memory) at the start of the 16-stream Q6, 21 min after boot. Not
  repeated: fresh boot + Q6 (min MemAvailable 4.30 GiB), and the same sequence again (min 3.75 GiB, 47 min up, no new NVRM errors).
  Cause not established; the patches were not A/B tested for it. Yesterday's gdn8k boot logged 32 NVRM errors in its first hour with
  none of these patches. Margin is thin in this flavor (GDN8 +2.1 GB on the same 22G pool). Added: `bp33` KV 22G -> 20G
  (exact, ~675k tokens, 2.6 contexts), then the same sequence + langcheck.

### 2026-11-04 11:14 UTC: calib control (pre-registered before running)
- calib top-1 excess: base33 +0.44 points (32.6k tokens, z +3.4); bp33 +0.31 (34.0k, z +2.3); bp33-large +0.43 (103k, z +5.9). So the
  fix did not shrink it, although the model-free script goes from 0.417 to 0.500. Control: same calib on `nospec` (bp33 without
  speculative-config, every token sampled directly). Reading: control z within +-2: a real bias remains in the speculative path.
  Control near +0.4: the test has a flaw (reported logprobs vs sampled distribution), and calib says nothing about #61161.

### 2026-11-04 09:40 UTC: pre-registration, backport from vLLM 0.33.0 (the user: "get the improvements from that new image and apply them to what we have today")
- Base: the live trial `gdn8k` on v5.4 (vLLM 0.32.0). Candidate `bp33` = gdn8k + mounted files in files/v33/:
  - #61161, block verification reuses random draws across rounds (V2 runner, block, T > 0: our exact config). Checked
    on this GPU with the issue's model-free script, 200k requests, P(token 2 = 0), exact 0.5: unpatched 0.4174, then
    0.4854 / 0.4688 / 0.4650; patched 0.5006, 0.4985 / 0.4976 / 0.4990.
  - #59511, DFlash context combine and anchor in the draft graph. #59260 + the DFlash hunk of #63819, warmup gaps
    (the first-use gap logged here). #62884, libgomp fork deadlock. #63165, SHM reader memory growth (~41 B per message).
  - #54163 by config, no code: `disable_eagle_block_drop` in speculative-config (DFlash KV does not depend on the
    next token, so the trailing prefix-cache block need not be dropped).
- Left out, with reasons: #61355 (non-speculative decode on SM100 family only, not our path); #62784 (conflicts in
  dp_utils, ~2 ms of a ~200 ms prefill-only step); #59973, #61601, #60050 (equivalent already here); #61213, #61996 (no
  masked-to-under-k case here; ~7 empty launches per step); #53987 FLy (lossy); #54448 XPress (no head for this model);
  #61343 (C++, not mountable).
- Measures, base on the live serve, then bp33 (`bp33_gates.sh`): calib.py (new: observed vs expected rank frequencies
  on the real model), exact_boot x2 (floor) and across, run_set quick, P3 nextturn, Q6 greedy, Q6 sampled at the serve's
  real sampling (T 1, top-p 0.95, top-k 20; 2 samples per task).
- Pass rule: speed: capacity, agents total_tps and prefill not more than 5 % worse, TTFT p90 not worse by more than
  10 %. Exactness: greedy divergence and KL at the base's own A/A floor (#61161 does not touch greedy). Quality: Q6
  greedy unchanged, Q6 sampled not more than 2 points below base, tool arguments valid, no CJK. calib: |z| of bp33
  not above base's. Fail on any one: drop the responsible patch, not the set.

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
