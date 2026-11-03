"""CPU: scale format v2 round trip and size on the real layer."""
import sys
import torch
sys.path.insert(0, "/kit/files")
import mbx_moek as M
mp = torch.load("/kit/cache/moek-layer10-marlin-plain.pt")
tot_p = tot_c = 0
for si, gu in ((1, True), (4, False)):
    s = mp[si]
    pk, off, ent = M.pack_scales(s, gu)
    back = M.unpack_scales(pk, off, ent, s.shape[1], s.shape[2])
    nexc = int(off[-1])
    comp = pk.numel() + off.numel() * 4 + nexc * 4
    tot_p += s.numel(); tot_c += comp
    print(f"lossless {torch.equal(back, s.view(torch.uint8))}, exceptions {nexc / s.numel() * 100:.3f} % of scales, "
          f"{comp / s.numel():.4f} bytes/scale")
print(f"layer: {tot_c / tot_p:.4f} of plain -> model {7.55 * tot_c / tot_p:.2f} GB")
