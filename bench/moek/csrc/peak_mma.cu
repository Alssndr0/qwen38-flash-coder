// mma.sync m16n8k16 bf16 -> fp32 throughput on this GPU (independent accumulators, no memory traffic).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
template <int ACC>
__global__ void peak(float* out, int iters) {
  uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u};
  uint32_t b[2] = {threadIdx.x * 11u, threadIdx.x * 13u};
  float c[ACC][4] = {};
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < ACC; i++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < ACC; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
void run(torch::Tensor out, int64_t blocks, int64_t threads, int64_t iters, int64_t acc) {
  auto st = c10::cuda::getCurrentCUDAStream();
  if (acc == 8) peak<8><<<blocks, threads, 0, st>>>(out.data_ptr<float>(), iters);
  else peak<16><<<blocks, threads, 0, st>>>(out.data_ptr<float>(), iters);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
