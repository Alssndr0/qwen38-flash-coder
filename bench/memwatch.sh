#!/usr/bin/env bash
# One line per second: time, MemAvailable GiB, swap used GiB, serve up (1/0). usage: memwatch.sh FILE SECONDS
end=$(( $(date +%s) + $2 ))
while [ $(date +%s) -lt $end ]; do
  awk -v t="$(date -u +%T)" -v up="$(docker ps -q --filter name=qwen38-flash-next --filter status=running | wc -l)" '/MemAvailable/{a=$2} /SwapTotal/{st=$2} /SwapFree/{sf=$2} END {printf "%s %.2f %.2f %s\n", t, a/1048576, (st-sf)/1048576, up}' /proc/meminfo >> "$1"; sleep 1
done
