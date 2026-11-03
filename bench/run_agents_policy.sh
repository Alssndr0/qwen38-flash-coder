#!/usr/bin/env bash
# run the agents scenario under live policies: run_agents_policy.sh NAME 'JSON' [NAME 'JSON' ...]
cd "$(dirname "$0")"
LIVE=../cache/mbx-smooth.json
while [ $# -ge 2 ]; do
  N=$1; J=$2; shift 2
  echo "$J" > $LIVE.tmp && mv $LIVE.tmp $LIVE; sleep 2
  echo "== $N $J $(date -u +%T)"
  python3 conc.py agents --trial "$N" --set quick > results/agents-$N.log 2>&1 || echo "FAILED $N"
done
echo '{"mode": "off", "trace": true}' > $LIVE.tmp && mv $LIVE.tmp $LIVE
echo "== done $(date -u +%T)"
