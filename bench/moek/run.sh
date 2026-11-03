#!/usr/bin/env bash
# Run a script from bench/moek in a side container of the serve image (the serve keeps running; keep it small).
#   bench/moek/run.sh SCRIPT.py [args]          CSF=0 for plain Marlin scales (default 1, as the serve)
#   NCU="--section ..." bench/moek/run.sh X.py  under Nsight Compute (host install mounted, needs SYS_ADMIN)
K="$(cd "$(dirname "$0")/../.." && pwd)"
extra=(); entry=python3
if [ -n "${NCU:-}" ]; then
  extra=(--cap-add SYS_ADMIN -v /opt/nvidia/nsight-compute/2025.3.1:/ncu:ro)
  entry=/ncu/ncu
  # shellcheck disable=SC2206
  set -- ${NCU} python3 "$@"
fi
name=moek-$$
trap 'docker kill "$name" >/dev/null 2>&1' EXIT INT TERM
docker run --rm --gpus all --ipc=host --name "$name" -e WHAT -e T -e UNIFORM -e SIGMA -e LAYER -e MOEK_VERBOSE -e CUDA_LAUNCH_BLOCKING -e MOEK_VARIANT -e MOEK_BM_PY -e ROWS -e MOEK_STREAMK_MAX_T -e MOEK_SPLIT_MAX_T -e MOEK_SPLIT_UNITS -e MOEK_ORDER -e MOEK_NS_GATE -e MOEK_NS_DOWN -e MOEK_NU -e TS -e MOEK_TILED -e MBX_MARLIN_CSF=${CSF:-1} -e HUMMING_CACHE_DIR=/kit/cache/humming \
  -e TRITON_CACHE_DIR=/kit/cache/triton-moek "${extra[@]}" -v "$K":/kit -w /kit/bench/moek --entrypoint "$entry" \
  myllmbox/qwen38-flash-next-vllm:v5.4 "$@"
