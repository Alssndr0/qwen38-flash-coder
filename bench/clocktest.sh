#!/usr/bin/env bash
# Step time vs GPU clock on the running serve. usage: clocktest.sh MHZ [MHZ ...]  Always restores the 2200 MHz cap on exit.
cd "$(dirname "$0")"
trap 'sudo -n nvidia-smi -lgc 300,2200 >/dev/null; echo "== restored cap 2200: $(nvidia-smi --query-gpu=clocks.max.graphics,clocks.current.graphics --format=csv,noheader)"' EXIT
python3 setlive.py trace=true >/dev/null
for f in "$@"; do
  sudo -n nvidia-smi -lgc 300,$f >/dev/null || { echo "clock change failed"; exit 1; }
  sleep 3; T=clk$f; python3 conc.py capacity --trial $T > /dev/null 2>&1
  for i in 1 2 3 4 5 6; do curl -s -m 30 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3.8-Flash-Next","messages":[{"role":"user","content":"OK"}],"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null; sleep 2; done
  echo "== $f MHz (seen $(nvidia-smi --query-gpu=clocks.current.graphics,temperature.gpu --format=csv,noheader)): $(python3 steps.py $T capacity | grep 'decode-only' | cut -c1-110)"
done
python3 setlive.py trace=- >/dev/null
