#!/usr/bin/env bash
# Does physically tidier memory speed up the memory-bound step? Compact + drop caches right before the boot, then measure.
cd "$(dirname "$0")"
docker stop -t 30 qwen38-flash-next >/dev/null; sleep 45
echo "before: $(tail -1 /proc/buddyinfo | cut -c1-110)"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'; sleep 20
echo "after:  $(tail -1 /proc/buddyinfo | cut -c1-110)"
./up.sh bp33 || exit 1
./clocktest.sh 2200 2>&1 | grep -v restored | cut -c1-170
echo "== frag done $(date -u +%T)"
