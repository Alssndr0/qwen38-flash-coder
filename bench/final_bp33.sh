#!/usr/bin/env bash
cd "$(dirname "$0")"; LAB=~/projects/llm-tuning-lab
export LAB_URL=http://127.0.0.1:8000 LAB_MODEL=Qwen/Qwen3.8-Flash-Next LAB_CONTAINER=qwen38-flash-next
./up.sh bp33 || { echo "== final boot FAILED"; exit 1; }
docker logs qwen38-flash-next 2>&1 | grep -m1 'GPU KV cache size' | cut -c50-140
echo "== $(date -u +%T) langcheck"; python3 $LAB/tests/quality/langcheck.py bp33 --model "$LAB_MODEL" --url "$LAB_URL" 2>&1 | tail -3 | cut -c1-160
./seq_q6.sh final-bp33
echo "== final done $(date -u +%T)"
