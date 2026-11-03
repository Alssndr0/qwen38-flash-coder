"""Cost of 4-bit-offset scale formats on real layer-10 scales (Marlin plain layout), CPU: bytes per scale for
block shapes (positions x k-groups) sharing one base, with per-scale exceptions (4 bytes each)."""
import torch
mp = torch.load("/kit/cache/moek-layer10-marlin-plain.pt")
tot_plain = 0
for name, s in (("w13", mp[1]), ("w2", mp[4])):
    b = s.view(torch.uint8).to(torch.int16)  # [E, K/16, N]
    e, kr, n = b.shape
    tot_plain += b.numel()
    for cols, kg in ((1, 8), (1, 10), (1, 16), (1, 20), (1, 40), (2, 8), (4, 8), (8, 8), (8, 4), (8, 10), (8, 20), (64, 1), (64, 8)):
        if kr % kg or n % cols:
            continue
        v = b.view(e, kr // kg, kg, n // cols, cols).permute(0, 1, 3, 2, 4).reshape(-1, kg * cols)  # blocks
        # best base: the window [m, m+15] with the most values, m among each block's values (approx: try min and
        # quantiles)
        best = None
        for q in (0.0, 0.02, 0.05, 0.1):
            m = torch.quantile(v.float(), q, dim=1, keepdim=True).floor().to(torch.int16) if q else v.min(1, keepdim=True).values
            out = ((v < m) | (v > m + 15)).sum(1)
            best = out if best is None else torch.minimum(best, out)
        exc = best.sum().item() / v.numel()
        cost = 0.5 + 1 / (kg * cols) + exc * 4
        print(f"{name} block {cols:2d} pos x {kg:2d} kg: exceptions {exc * 100:6.3f} % of scales, bytes/scale {cost:.4f}")
