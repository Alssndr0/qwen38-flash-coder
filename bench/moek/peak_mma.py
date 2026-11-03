import torch
from torch.utils.cpp_extension import load
m = load(name="peak_mma", sources=["csrc/peak_mma.cu"], build_directory="/kit/cache/moek-build-peak",
         extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"])
import os; os.makedirs("/kit/cache/moek-build-peak", exist_ok=True)
for threads, acc in ((128, 16), (256, 8), (256, 16), (512, 8)):
    blocks, iters = 48 * 8, 2000
    out = torch.empty(blocks * threads, device="cuda")
    m.run(out, blocks, threads, 10, acc); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); m.run(out, blocks, threads, iters, acc); b.record(); torch.cuda.synchronize()
    ms = a.elapsed_time(b)
    flops = blocks * threads / 32 * iters * acc * 2 * 16 * 8 * 16
    print(f"threads {threads} acc {acc}: {flops / ms / 1e9:.1f} TFLOP/s")
