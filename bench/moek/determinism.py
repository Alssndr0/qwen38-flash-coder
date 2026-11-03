"""Same inputs twice (plain, packed) and packed vs plain: bitwise, at several sizes."""
import torch, kcuda, problem as P
mp = P.marlin_plain(10)
mpp = kcuda.tile_layout(kcuda.with_packed(mp))
for t in (6912, 2000, 128, 64, 8, 1):
    x = (torch.randn((t, P.HID), device=P.DEV) * 0.5).to(torch.bfloat16)
    ids, wts = P.routing(t, sigma=0.6)
    a1, a2 = kcuda.moe(x, ids, wts, mp), kcuda.moe(x, ids, wts, mp)
    b1, b2 = kcuda.moe(x, ids, wts, mpp), kcuda.moe(x, ids, wts, mpp)
    print(f"T={t}: plain repeat {torch.equal(a1, a2)}, packed repeat {torch.equal(b1, b2)}, packed == plain {torch.equal(a1, b1)}")
