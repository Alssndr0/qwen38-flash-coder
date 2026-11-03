# Qwen Flash Coder: v4gdnq

Qwen3.8-Flash-Next tuned for coding in English.

## Run it

On a DGX Spark with Docker and the NVIDIA container runtime (`./tune-host.sh` once per box):

```bash
git clone https://github.com/Alssndr0/qwen38-flash-coder.git
cd qwen38-flash-coder
./run.sh
```

The first run downloads the AZ checkpoint at the measured revision (about 100 GB) and the DFlash2 drafter (1.2 GB,
sha256-checked), downloads this repo's release assets (4.7 GB, sha256-checked), assembles the checkpoint
(AZ's files are hard-linked, only shard 1 is new), prepares the n-gram table map (once), and serves an
OpenAI-compatible API on `http://127.0.0.1:8000/v1` as `Qwen/Qwen3.8-Flash-Next`. The first boot also compiles the
kit's expert kernel (about 1.5 minutes, cached in `cache/moek-build`). Later boots take about 4 to 5 minutes.

- `PREPARE_ONLY=1 ./run.sh` fetches and assembles, and touches no container.
- `DRY_RUN=1 ./run.sh` prints the `docker run` command and changes nothing.
- `./stop.sh` stops the serve, `./view.sh` follows its logs.
- Everything is configured in [recipe.yaml](recipe.yaml); lines that differ from myllmbox's v5.4 recipe are marked
  `[coder]`.

The serve needs about 105 GiB of free GPU memory to boot (vLLM's start check at `gpu-memory-utilization` 0.86; on the
Spark the page cache does not count as free). The weights take 67.8 GiB and the KV pool 22 GB (fp8, 742,741 tokens: 2.8
full 262k contexts), which leaves room for builds and containers next to the serve.

## What it is

| Part | Choice |
|---|---|
| Checkpoint | [azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ) at `5d927ea`: NVIDIA FP4-Local routed at 5 experts per token, shared expert widened and healed |
| GDN projections | in_proj / out_proj of every linear-attention layer in NVFP4 (W4A16), GPTQ with recorded Hessians; held-out output error 0.0345 (round-to-nearest 0.0599) |
| Routed experts | NVFP4 as shipped, on the kit's own CUDA kernel (`files/mbx_moek.cu`) with its own lossless compressed scales; the image's Marlin with `MBX_MOEK: "0"` |
| Other dense layers | MXFP8 as shipped (attention, shared expert) |
| Hyper-connections | INT8 per-128 groups with offline GPTQ codes (`gptq_hc_int8_g128.pt`) |
| Output head | vocabulary trimmed to 144,130 Latin-script tokens, INT8; the other rows score -inf |
| Drafter | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) at `a22a174`, block 8, fixed depth 7, drafter linears in NVFP4, shortlist head |
| Prefix cache | 64-token matching, also across the drafter's sliding-window layers (`DFL2_SWA_AS_FULL`) |
| Engine | [myllmbox/qwen38-flash-next-vllm:v5.4](https://hub.docker.com/r/myllmbox/qwen38-flash-next-vllm) (vLLM 0.32), measured image `sha256:67f00fec76da`, plus the files in [files/](files/) mounted read-only |
| Context | native 262,144 tokens (no YaRN) |

Every drafted token is verified by the target, so the drafter changes speed, not outputs.

## Results so far

Speed, one DGX Spark:

| Measure | v4gdnq | f5c (same stack, MXFP8 GDN) | hibrid48 production |
|---|---|---|---|
| Decode step, 1 stream | 41.8 ms | 45.8 ms | |
| Agentic replay, quick set | 84.1 tok/s | | |
| Agentic replay, full set (177 turns) | | 81.3 tok/s | 68.1 tok/s |
| Decode per request, 3 agents at once | 53.7 tok/s | 50.6 tok/s | |

Measured 2026-11-01 with the image's Marlin for the experts; the kit's kernel leaves decode step times equal or up to
6 % lower ([The expert layers](#the-expert-layers)).

Quality:

| Measure | v4gdnq | f5c | FP4-Local, 10 experts |
|---|---|---|---|
| Code eval (EvalPlus) | 453 | 451 | |
| Held-out NLL (lower is better) | 1.0215 | 1.0254 | 1.0275 (reference serve) |
| Fresh tasks, half A, 40-min cap (of 38) | 16 | 16 | 13 (16 at 100 min) |

The fresh tasks are real bug fixes merged on or after 2026-09-15, in Django, pydantic, pytest, Prometheus, Terraform,
Caddy and etcd, solved by OpenCode at xhigh effort with no internet, and scored by each PR's own hidden tests. On the
same tasks v4gdnq and f5c each solved one task the other missed.

## Smooth with several agents at once

With one request at a time nothing changes. With several, vLLM runs a newcomer's prompt in chunks of up to 6,912
tokens, and every such step streams most of the 4-bit expert weights: the other streams freeze for 0.3 s (a small
turn) to 2.5 s per chunk. The kit's scheduler (`files/mbx_sched.py`) and worker hook (`files/mbx_sidecar.py`) run the
newcomer's prompt a few of the model's 48 layers per step instead, right after the normal decode step, sized so the
decoding streams keep about 220 ms between deliveries. The prompt's first token comes back in the step that finishes
its last layer. Outputs are those of a normal prefill: next-token KL at the run-to-run floor, including later turns
that reuse a sliced turn's prefix cache (`bench/exact2.py`, `bench/exact3.py`).

4 agents replaying 62 recorded OpenCode / Pi turns (`bench/conc.py agents`), one DGX Spark:

| Measure | vLLM's chunked prefill | Layer slicing (2026-11-02) | Now: + expert kernel, refitted planner (2 runs) |
|---|---|---|---|
| Decode time lost to stalls (gaps beyond 250 ms) | 12.7 % | 0.75 % | 0.64 / 0.49 % |
| Gaps over 1 s, per 1,000 tokens | 1.05 | 0 | 0 / 0 |
| Worst gap / 99th percentile gap | 3.04 s / 588 ms | 0.70 s / 258 ms | 0.69 s / 258 ms, 0.40 s / 264 ms |
| Decode speed per stream | 38.9 tok/s | 44.1 tok/s | 41.7 / 41.7 tok/s |
| Time to first token, median (turns under 1k new tokens) | 0.88 s (0.66 s) | 1.49 s (0.97 s) | 1.33 / 1.34 s (not split) |
| Total, 4 agents | 114.6 tok/s | 105.1 tok/s | 112.0 / 109.8 tok/s |

The replays' output lengths differ from run to run (the agents take different paths), so totals move by a few
percent between runs of the same serve. The price of slicing is the newcomer's wait: its prefill shares each step
with the others' decoding. The policy is the `smooth:` section of `recipe.yaml`; `run.sh` writes it to
`cache/mbx-smooth.json`, which the serve re-reads while it runs (`"mode": "off"` is vLLM's own chunked prefill, no
restart needed). Requests with logprobs, images or a grammar are
not sliced. Each slice gets as many layers as fit the gap target under a per-layer cost model, `la + lb * tokens` ms
(`la`, `lb` in `smooth:`), fitted on the serve's own slice log with `bench/lbfit.py`; refit it after any change that
makes layers cheaper. It was 2.4 / 0.0071 with the image's Marlin and is 2.9 / 0.0066 with the expert kernel: with the
old values a full chunk's layer was overestimated by 6 % and slices often got one layer fewer than the target allows
(newcomers now 4-9 % faster, `bench/DECISIONS.md`). The measuring tools are in `bench/` (README there).

### Faster prefill (opt-in, changes prefill outputs slightly)

The GDN projections are 4-bit (GPTQ) because that is fastest for decode; on a prefill batch Marlin's 4-bit GEMM is
compute-bound at about half the tensor-core rate of MXFP8. `files/mbx_gdn8.py` keeps the 4-bit weights for decode and
loads the AZ checkpoint's MXFP8 originals of the same projections (+2.1 GB); batches of 256 tokens or more run on them.
Decode (at most 128 tokens per step) is unchanged. Measured paired on one boot, with the image's Marlin for the
experts (`bench/DECISIONS.md`, 2026-11-03):

| Measure | 4-bit everywhere | MXFP8 for prefill |
|---|---|---|
| Prefill alone, 10k / 23k / 66k tokens | 3.73 / 8.42 / 24.13 s | 3.56 / 7.95 / 22.77 s |
| Newcomer wait with 3 streams decoding, 10k / 40k | 6.17 / 22.83 s | 5.48 / 21.28 s |
| 4 agents: stalls, decode per stream, TTFT p90 | 0.75 %, 40.5 tok/s, 5.52 s | 0.74 %, 42.9 tok/s, 4.45 s |
| Code eval (EvalPlus, of 542) | 453 | 453 |
| Held-out NLL | 1.0164 | 1.0247 (f5c 1.0254) |

Stacked with the expert kernel (boot `bench/trials/gdn8k.yaml`): prefill alone 3.51 / 7.88 / 13.91 / 22.50 s ->
3.30 / 7.41 / 13.23 / 21.38 s (fresh 10k / 23k / 40k / 66k, 5-6 % faster), model memory 69.9 GiB. Turn it on with the
commented `MBX_GDN8` lines in `recipe.yaml` and set `la: 3.0`, `lb: 0.0062` in `smooth:` (fitted on that boot's
slices), or boot the trial as it is: `RECIPE=bench/trials/gdn8k.yaml ./run.sh`. `cache/mbx-gdn8.json` switches it
live: `{"min_tokens": 256}` on, `{"min_tokens": 100000000}` off (the file wins over the environment).

### The expert layers

The routed experts (512 per layer, 5 per token, NVFP4 weights) run on the kit's own CUDA kernel, `files/mbx_moek.cu`
(`MBX_MOEK` in `recipe.yaml`; `files/mbx_moek.py` builds it into `cache/moek-build` on the first boot and swaps it in
for vLLM's Marlin MoE). The image's compressed-scale Marlin decodes the 4-bit weights' scales once per 48-row block,
which costs about a fifth of a prefill layer's expert time; plain Marlin is faster there but needs 3.2 GB more for
the scales. The kernel reads its own compressed scales (lossless, 3.93 GB for the model, 0.4 GB less than the
image's) and decodes them once per weight tile, takes all the rows of an expert in one pass over its weights, streams
the weights from a dedicated copy warp, fuses `silu_and_mul`, and reorders the weights once at load so that each
pipeline stage is one contiguous 8 KB copy. Small decode batches split the work along k so every SM streams the same
expert at the same time.

| one layer, real weights (`bench/moek`) | image's Marlin | kernel |
|---|---|---|
| prefill chunk, 6,912 tokens | 12.2 ms | 9.6 ms |
| decode, 8 / 16 / 32 / 64 / 128 tokens | 0.47 / 0.82 / 1.56 / 2.56 / 3.72 ms | +0.9 / -0.1 / -1.6 / -2.1 / -2.8 % |

In the serve: prefill alone 3.67 / 8.26 / 14.58 / 23.61 s -> 3.51 / 7.88 / 13.91 / 22.50 s (fresh 10k / 23k / 40k /
66k); decode step times at 1-4 streams equal or 1-6 % lower; the stall scenario and the agents set unchanged or
slightly better (newcomer TTFT 1-7 % lower). Outputs: the same arithmetic as Marlin summed in a different order; the
error against a bf16 reference is equal or lower at every size, and greedy decode against the previous serve stays
within that serve's own run-to-run spread (`bench/exact_boot.py`). Model memory after load is 67.8 GiB (68.2 with the
image's Marlin). `MBX_MOEK: "0"` goes back to the image's Marlin,
where `files/mbx_moe.py` keeps its row tile at 48 for prefill batches (prefill 1.6-2.8 % faster than vLLM's 64).
Tried and not taken (`bench/DECISIONS.md`): the Humming MoE kernels (prefill 5-6 % faster, decode 6-7 % slower),
Triton prototypes (no faster than Marlin), stream-k scheduling (balanced, but the SMs then stop sharing activation rows
in L2), and larger prefill chunks (newcomers 4-13 % faster but the decoding streams' worst freeze up to a third longer).

## Files

| Path | What it is |
|---|---|
| `run.sh` | fetch, check, assemble, then `serve.sh` |
| `serve.sh` | the launcher the measurements were taken with (myllmbox's v5.4 `run.sh` plus `RECIPE` / `EXTRA_DOCKER` / `DRY_RUN`) |
| `recipe.yaml` | the full serve configuration |
| `files/dfl2_opt.py` | drafter in NVFP4 with the shortlist head; trimmed INT8 target head (chunked for long prompts); the sliding-window fix for 64-token prefix hits |
| `files/v54/` | the DFlash2 hidden-state capture, rebased onto v5.4's Qwen4Exp model and GEMM files |
| `files/dflash2/` | DFlash2 support in vLLM's config check and speculator |
| `files/base_loader.py`, `files/mbx_opt_quant.py` | loader hook and INT8 hyper-connection quantization with GPTQ codes |
| `files/dfl2_basis_card_r256.pt` | the drafter's shortlist basis |
| `files/vocab_keep_latin.json` | the 144,130 vocabulary rows the head keeps |
| `files/mbx_sched.py` | scheduler: vLLM's async scheduler plus step tracer, prefill pacing and layer slicing (live policy) |
| `files/mbx_sidecar.py` | worker hook that runs the sliced prefill (with `_forward_slice` in `files/v54/model.py`) |
| `files/mbx_gdn8.py` | opt-in: MXFP8 GDN projections for prefill-size batches, 4-bit for decode (hook in `files/base_loader.py`) |
| `files/mbx_moek.py`, `files/mbx_moek.cu` | the routed experts' CUDA kernel and its vLLM hook (weights and scales converted at load; hook in `files/base_loader.py`) |
| `files/mbx_moe.py` | Marlin row tile for the routed experts on prefill-size batches, used when `MBX_MOEK` is off (hook in `files/base_loader.py`) |
| `files/mounts.txt` | where each file goes in the image |
| `model/` | the assembled checkpoint's `config.json`, conversion error reports and shard 1 checksum |
| `assets.sha256` | checksums of the release assets |
| `build/` | how the GDN shard was made (below) |
| `bench/` | concurrency and smoothness measurements: scenarios, report, exactness and robustness checks, profiling, the slice cost fit; `bench/moek/` is the expert kernel's test bench, `bench/trials/` the trial recipes, `bench/DECISIONS.md` the log of every change and its measurements |

Release `v4gdnq-2026-11-01`: shard 1 of the checkpoint in three parts, its weight index (gzip), and the INT8 GPTQ codes.

## How the GDN shard was made

1. Boot `build/calgdn.yaml` (the AZ checkpoint with the recorder `build/mbx_opt_quant_cal.py`) and drive it with
   `build/calib_drive.py` over the calibration prompts, recording a Hessian per GDN projection.
2. `build/make_fp4_variant.py --gptq <Hessian dir>` converts those projections from MXFP8 to NVFP4 W4A16 (one shared
   global scale per fused group) and writes the new shard 1, index and config.

The paths in `build/` are the ones of the original workspace, kept as a record.

## License

Code: Apache License 2.0 ([LICENSE](LICENSE)), except the files taken from myllmbox's recipe, which stay under their
authors' terms. Weights (the release assets and `model/config.json`): derived from NVIDIA's Qwen3.8-Flash-Next-FP4-Local
via azampatti's AZ checkpoint, under the NVIDIA Open Model Agreement
([LICENSE.NVIDIA-Open-Model-Agreement.txt](LICENSE.NVIDIA-Open-Model-Agreement.txt)) and the Qwen Community License 1.0
([LICENSE.Qwen-Community-License-1.0.txt](LICENSE.Qwen-Community-License-1.0.txt)). Details in [NOTICE](NOTICE).
