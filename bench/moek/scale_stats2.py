"""Range of real scale bytes along k (same column, consecutive 16-k groups), Marlin layout."""
import torch
import problem as P
w13, w2 = P.real_weights(10)
mp = P.marlin_prepare(w13, w2)
for name, s in (("w13", mp[1]), ("w2", mp[4])):
    b = s.view(torch.uint8).to(torch.int16)  # [E, K/16, N]
    e, kr, nn = b.shape
    for g in (4, 8, 10, 16, 20, 40):
        if kr % g:
            continue
        v = b.view(e, kr // g, g, nn)
        rng = v.max(2).values - v.min(2).values
        for lim in (15, 31):
            pass
        print(f"{name} K/16={kr} group {g:2d} along k: range<=7 {(rng <= 7).float().mean() * 100:7.3f} %, <=15 "
              f"{(rng <= 15).float().mean() * 100:7.3f} %, <=31 {(rng <= 31).float().mean() * 100:7.3f} %, max {int(rng.max())}; "
              f"bytes/scale at 4 bit + base {0.5 + 1 / g:.3f}")
