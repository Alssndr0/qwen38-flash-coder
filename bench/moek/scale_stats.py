"""Marlin-layout scale bytes of real layers: value range per segment, to size a lossless 4-bit-offset format."""
import torch
import problem as P
for layer in (10,):
    w13, w2 = P.real_weights(layer)
    mp = P.marlin_prepare(w13, w2)
    for name, s, n in (("w13", mp[1], 1280), ("w2", mp[4], 2560)):
        b = s.view(torch.uint8).to(torch.int16)  # [E, K/16, N]
        e, kr, nn = b.shape
        print(f"layer {layer} {name}: shape {tuple(b.shape)}, byte min {int(b.min())} max {int(b.max())}, zero frac {(b == 0).float().mean():.5f}")
        for seg in (16, 64, 128):
            v = b.view(e, kr, nn // seg, seg)
            nz = v.clone(); nz[nz == 0] = 999
            rng = (v.max(-1).values - nz.min(-1).values.clamp(max=255))
            rng_all = v.max(-1).values - v.min(-1).values
            print(f"   seg {seg:3d}: range<=15 {(rng_all <= 15).float().mean() * 100:.3f} %  (ignoring zeros {(rng <= 15).float().mean() * 100:.3f} %), "
                  f"range<=31 {(rng_all <= 31).float().mean() * 100:.3f} %, max range {int(rng_all.max())}")
