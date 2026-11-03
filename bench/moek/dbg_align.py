import torch, problem as P, kcuda
t = 6912
ids, wts = P.routing(t, sigma=0.6)
sid, eid, npad = kcuda.align(ids, P.E)
n = int(npad[0]); nb = n // 128
s = sid[:n].view(nb, 128)
valid = (s < t * P.TOPK)
nv = valid.sum(1)
last = torch.where(valid, torch.arange(128, device=s.device) + 1, 0).max(1).values
print("blocks", nb, "sorted_ids len", sid.numel(), "pad value", int(sid[-1]), "sum ceil(nvalid/16)", int(((nv + 15) // 16).sum()), "sum ceil(last/16)", int(((last + 15) // 16).sum()))
print("first block ids", s[0, :8].tolist(), s[0, -8:].tolist(), "eid", eid[:4].tolist())
