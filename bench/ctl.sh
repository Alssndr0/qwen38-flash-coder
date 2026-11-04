#!/usr/bin/env bash
cd "$(dirname "$0")"
./up.sh nospec || exit 1
docker logs qwen38-flash-next 2>&1 | grep -c -i 'SpecDecoding\|speculative_config=None' 
python3 calib.py nospec-large --n 180 --c 4
echo "== ctl done $(date -u +%T)"
