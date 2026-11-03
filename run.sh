#!/usr/bin/env bash
# Qwen Flash Coder (v4gdnq) on ONE DGX Spark: fetch and check everything, assemble the checkpoint, then serve with
# serve.sh (the launcher the measurements were taken with) on recipe.yaml. OpenAI API on :8000.
#   ./run.sh                 fetch if needed, assemble, serve (replaces any running qwen38-flash-next container)
#   PREPARE_ONLY=1 ./run.sh  fetch and assemble only; touches no container
#   DRY_RUN=1 ./run.sh       print the docker command; downloads and changes nothing
# Needs: docker with the NVIDIA runtime, curl, ~110 GB of disk.
set -euo pipefail
cd "$(dirname "$0")"
K="$(pwd)"

AZ_REPO=azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ
AZ_REV=5d927eabd4e4766a69bf12f7a1387b01d936a78f            # the revision measured
DRAFTER=incoai/Qwen3.8-Flash-Next-DFlash2
DRAFTER_REV=a22a1744e85c17364dcc54a6a420af7135d4c7f6
DRAFTER_SHA=d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f
RELEASE_REPO=Alssndr0/qwen38-flash-coder
RELEASE_TAG=v4gdnq-2026-11-01
IMAGE="$(awk '$1=="image:"{print $2; exit}' recipe.yaml)"
AZ="models/Qwen3.8-Flash-Next-FP4-Local-AZ"; W="models/Qwen3.8-Flash-Next-FP4-Local-AZ-w4gdnq"; D="models/Qwen3.8-Flash-Next-DFlash2"
mkdir -p models cache logs .assets

hf_get() {  # hf_get <repo> <revision> <local dir> [allow patterns as a python list]
  docker run --rm -e HF_TOKEN -v "$K/models:/dl" --entrypoint python3 "$IMAGE" -c "
from huggingface_hub import snapshot_download; import subprocess
snapshot_download('$1', revision='$2', local_dir='/dl/$(basename "$3")', allow_patterns=${4:-None})
subprocess.run(['chown', '-R', '$(id -u):$(id -g)', '/dl/$(basename "$3")'], check=False)"
}
need() { [ -z "${DRY_RUN:-}" ] || { echo "· (dry run) $1 is missing: a real run would fetch it"; return 1; }; }

# 1. the AZ checkpoint (NVIDIA FP4-Local routed at 5 experts, healed shared expert; ~100 GB)
if [ ! -f "$AZ/model.safetensors.index.json" ] || [ -n "$(find "$AZ/.cache" -name '*.incomplete' -print -quit 2>/dev/null)" ]; then
  if need "$AZ"; then echo "· downloading $AZ_REPO@${AZ_REV:0:7} -> $AZ (resumable)"; docker pull -q "$IMAGE" >/dev/null; hf_get "$AZ_REPO" "$AZ_REV" "$AZ"; fi
fi

# 2. the DFlash2 drafter (1.2 GB), sha256-checked
if [ ! -f "$D/model.safetensors" ] && need "$D"; then
  echo "· downloading $DRAFTER@${DRAFTER_REV:0:7} -> $D"; hf_get "$DRAFTER" "$DRAFTER_REV" "$D" "['config.json', 'model.safetensors']"
fi
if [ -f "$D/model.safetensors" ] && [ ! -f "$D/.sha256-ok" ]; then
  got="$(sha256sum "$D/model.safetensors" | cut -d' ' -f1)"
  [ "$got" = "$DRAFTER_SHA" ] || { echo "✗ $D/model.safetensors sha256 $got, expected $DRAFTER_SHA"; exit 1; }
  touch "$D/.sha256-ok"; echo "· drafter: sha256 ok"
fi

# 3. this repo's release assets (the 4-bit GDN shard in parts, its weight index, the INT8 GPTQ codes), sha256-checked
if ! (cd .assets && sha256sum --quiet -c ../assets.sha256 2>/dev/null); then
  if need "release assets"; then
    echo "· fetching release $RELEASE_TAG assets -> .assets/ (4.7 GB, resumable)"
    while read -r sum name; do
      (cd .assets && echo "$sum  $name" | sha256sum --quiet -c 2>/dev/null) && continue
      curl -fL --retry 5 -C - -o ".assets/$name" "https://github.com/$RELEASE_REPO/releases/download/$RELEASE_TAG/$name" \
        || { echo "✗ download of $name failed (rerun to resume)"; exit 1; }
    done < assets.sha256
    (cd .assets && sha256sum --quiet -c ../assets.sha256) || { echo "✗ release assets fail their sha256: delete the bad file in .assets/ and rerun"; exit 1; }
  fi
fi

# 4. assemble the checkpoint: AZ's files hard-linked (no copy), shard 1 / weight index / config.json replaced
if [ ! -f "$W/.complete" ] && [ -z "${DRY_RUN:-}" ]; then
  echo "· assembling $W"
  rm -rf "$W"; mkdir -p "$W/tensors"
  (cd "$AZ" && find . -type f ! -path './.cache/*' ! -name config.json ! -name model.safetensors.index.json \
     ! -name model-00001-of-00021.safetensors ! -name .gitattributes) | while read -r f; do
    mkdir -p "$W/$(dirname "$f")"; ln -f "$AZ/$f" "$W/$f" 2>/dev/null || cp "$AZ/$f" "$W/$f"
  done
  cat .assets/w4gdnq-model-00001-of-00021.safetensors.part* > "$W/tensors/model-00001-of-00021.safetensors"
  (cd "$W/tensors" && sha256sum --quiet -c "$K/model/shard1.sha256") || { echo "✗ assembled shard 1 fails its sha256"; exit 1; }
  gunzip -c .assets/w4gdnq-model.safetensors.index.json.gz > "$W/model.safetensors.index.json"
  cp model/config.json model/fp4_conversion_errors.json model/gptq_output_errors.json "$W/"
  touch "$W/.complete"; echo "  ✓ checkpoint assembled (shard 1 sha256 ok)"
fi

# 5. cache: the INT8 GPTQ codes for the hyper-connection layers, the drafter's live file (top-p / tau 1.0 = off)
if [ -z "${DRY_RUN:-}" ]; then
  [ -f cache/gptq_hc_int8_g128.pt ] || ln -f .assets/gptq_hc_int8_g128.pt cache/ 2>/dev/null || cp .assets/gptq_hc_int8_g128.pt cache/
  [ -f cache/dfl2-live.json ] || echo '{"top_p": 1.0, "tau": 1.0}' > cache/dfl2-live.json
fi
[ -n "${PREPARE_ONLY:-}" ] && { echo "✓ prepared (PREPARE_ONLY): ./run.sh serves it"; exit 0; }
[ -n "${DRY_RUN:-}" ] && [ ! -f "$W/.complete" ] && { echo "· (dry run) $W is not assembled yet: PREPARE_ONLY=1 ./run.sh first"; exit 0; }

# 6. serve: the files in files/mounts.txt go over the image's own copies, read-only
EXTRA=""
while read -r src dst; do
  case "$src" in ''|'#'*) continue;; esac
  EXTRA+=" -v $K/$src:$dst:ro"
done < files/mounts.txt
RECIPE="${RECIPE:-recipe.yaml}" EXTRA_DOCKER="$EXTRA ${EXTRA_DOCKER:-}" exec ./serve.sh
