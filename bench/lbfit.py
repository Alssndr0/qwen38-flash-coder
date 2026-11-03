"""Fit the slice planner's per-layer cost ms = la + lb * tokens from mbx_sidecar "tlog" lines (worker log)."""
import ast
import re
import sys
pat = re.compile(r"slice GPU ([\d.]+) ms CPU ([\d.]+) ms: job \S+ layers (\d+)\.\.(\d+) of (\d+) reqs (.*)$")
xs, ys, rows = [], [], []
for line in open(sys.argv[1]):
    m = pat.search(line)
    if not m:
        continue
    gpu, cpu, lo, hi, nl = float(m[1]), float(m[2]), int(m[3]), int(m[4]), int(m[5])
    reqs = ast.literal_eval(m[6])
    n = sum(r[1] for r in reqs)
    k = hi - lo
    last = hi == nl
    rows.append((n, k, gpu, cpu, last))
    if not last and k > 0:  # the last slice also runs the head / sampling: fitted apart
        xs.append(n)
        ys.append(gpu / k)
N = len(xs)
mx, my = sum(xs) / N, sum(ys) / N
lb = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
la = my - lb * mx
res = [y - (la + lb * x) for x, y in zip(xs, ys)]
rms = (sum(r * r for r in res) / N) ** 0.5
print(f"{N} slices (not last): per layer ms = {la:.2f} + {lb:.5f} * tokens   (rms {rms:.2f} ms per layer)")
for lo_n, hi_n in ((0, 2000), (2000, 5000), (5000, 7000), (7000, 20000)):
    sel = [(x, y) for x, y in zip(xs, ys) if lo_n <= x < hi_n]
    if sel:
        print(f"  tokens {lo_n}-{hi_n}: {len(sel)} slices, mean {sum(y for _, y in sel) / len(sel):.2f} ms/layer at "
              f"{sum(x for x, _ in sel) / len(sel):.0f} tokens; model 0.0071 -> {2.4 + 0.0071 * sum(x for x, _ in sel) / len(sel):.2f}")
lasts = [(n, k, g) for n, k, g, c, l in rows if l]
print(f"last slices: {len(lasts)}, layers {[k for _, k, _ in lasts][:12]}, GPU ms {[round(g) for _, _, g in lasts][:12]}")
