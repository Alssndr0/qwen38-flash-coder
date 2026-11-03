"""GB10 tensor-core peak per input type (mma.sync, Humming's micro-benchmark): the ceiling for an expert GEMM."""
from humming.ops.bench import tops_bench
for dt in ("bfloat16", "float16", "float8e4m3"):
    print(f"{dt:12s} {tops_bench(dt):7.1f} TFLOP/s (fp32 accumulate)")
print(f"{'float16':12s} {tops_bench('float16', use_f16_accum=True):7.1f} TFLOP/s (fp16 accumulate)")
