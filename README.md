# Qwen Flash Coder: v4gdnq

Qwen3.8-Flash-Next tuned for English agentic coding on one DGX Spark. This is the fastest stack of the 2026-10-30
search that has kept full quality so far: about 9 % faster per decode step than the previous best (f5c, itself 19 %
above the hibrid48 production serve on the full agentic replay), with the same results on hard, post-training coding
tasks.

Status (2026-11-01): **candidate**. The hard-task comparison (fresh bug-fix tasks, SWE-bench Multilingual, code review;
two halves) is still running; numbers below are what is measured so far.

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
OpenAI-compatible API on `http://127.0.0.1:8000/v1` as `Qwen/Qwen3.8-Flash-Next`. Later boots take about 4 minutes.

- `PREPARE_ONLY=1 ./run.sh` fetches and assembles, and touches no container.
- `DRY_RUN=1 ./run.sh` prints the `docker run` command and changes nothing.
- `./stop.sh` stops the serve, `./view.sh` follows its logs.
- Everything is configured in [recipe.yaml](recipe.yaml); lines that differ from myllmbox's v5.4 recipe are marked
  `[coder]`.

The serve needs about 100 GB of free memory to boot. The KV pool is 22 GB (fp8, 742,741 tokens: 2.8 full 262k contexts), which
leaves room for builds and containers next to the serve.

## What it is

| Part | Choice |
|---|---|
| Checkpoint | [azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ) at `5d927ea`: NVIDIA FP4-Local routed at 5 experts per token, shared expert widened and healed |
| GDN projections | in_proj / out_proj of every linear-attention layer in NVFP4 (W4A16), GPTQ with recorded Hessians; held-out output error 0.0345 (round-to-nearest 0.0599) |
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

Quality:

| Measure | v4gdnq | f5c | FP4-Local, 10 experts |
|---|---|---|---|
| Code eval (EvalPlus) | 453 | 451 | |
| Held-out NLL (lower is better) | 1.0215 | 1.0254 | 1.0275 (reference serve) |
| Fresh tasks, half A, 40-min cap (of 38) | 16 | 16 | 13 (16 at 100 min) |

The fresh tasks are real bug fixes merged on or after 2026-09-15, in Django, pydantic, pytest, Prometheus, Terraform,
Caddy and etcd, solved by OpenCode at xhigh effort with no internet, and scored by each PR's own hidden tests. On the
same tasks v4gdnq and f5c each solved one task the other missed.

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
| `files/mounts.txt` | where each file goes in the image |
| `model/` | the assembled checkpoint's `config.json`, conversion error reports and shard 1 checksum |
| `assets.sha256` | checksums of the release assets |
| `build/` | how the GDN shard was made (below) |

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
