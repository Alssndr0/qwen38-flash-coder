// W4A16 routed-expert GEMMs (NVFP4 weights, bf16 activations) on GB10 (sm_121), reading the serve's Marlin weight
// layout; one kernel for prefill and decode sizes.
//
// Persistent stream-k: one CTA per SM; every work item (moe_align block of up to BM rows of one expert x 256 output
// columns) is a run of K/64 k-tile positions, and each CTA takes one equal contiguous range of all positions (items
// cut at a range end are summed by the last of their CTAs), so the load is even at every batch size.
//   - producer warp (warp 8): streams every item's weights by bulk copy (TMA) into a WST-slot ring (full/empty
//     mbarriers) and runs ahead into the next item, so an SM's DRAM stream does not stop between items; with packed
//     scales it also copies one compressed run pair per 2 stages and decodes each stage's scales into the slot
//     (+ exceptions) LAG stages behind the issue ("decoded" mbarrier);
//   - 8 consumer warps: 4 column slabs x 2 row groups; per 16-k step a warp dequantizes its 8 mma B fragments
//     (Marlin's bit tricks, bf16 multiply by the scale) and runs 8 MMAs per activation fragment; the number of valid
//     16-row fragments (NF) is a compile-time variant, so padding rows cost no MMA; activations by cp.async
//     AD stages ahead;
//   - gate_up: each warp holds 32 gate and the matching 32 up columns, so the epilogue writes silu(gate) * up from
//     registers (vLLM's silu_and_mul arithmetic on the bf16-rounded results); down: rows [id, N] times the router
//     weight (applied with the global scale to the fp32 sum, as Marlin).
//
// Layouts (vLLM Marlin):
//   W: int32 [E, K/16, N*2]: per 16(k) x 64(n) tile 128 int32, tiles k-major; thread t's int4 at t*4 holds, in int g,
//      the 8 e2m1 codes of its mma B fragments for n16 group g (cols g*16 + t/4 and g*16 + 8 + t/4).
//   S (plain): uint8 [E, K/16, N]: Marlin S0E5M3 scales, permuted within 64 columns so thread t's 8 scales are the
//      8 bytes at 8*(t/4) (Marlin's dequant_fp8_scales order). Packed (kcuda.pack_scales): per (expert, run of KG
//      16-k groups, 128 columns) 128 bases + KG x 64 bytes of 4-bit offsets, exceptions in a side list.
//   G: fp32 [E]: global scale processed for the no-bias dequant (nvfp4_marlin_process_global_scale / scale factor).
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace moek {

constexpr int BM = 96;                  // rows per item (moe_align block); 9 warps leave 168 registers per thread
constexpr int BN = 256;                 // output columns per item (four Marlin 64-col tiles)
constexpr int BK = 64;                  // k per pipeline stage (four Marlin 16-k tiles)
constexpr int CTHREADS = 256;           // 8 consumer warps
constexpr int THREADS = CTHREADS + 32;  // + 1 producer warp
#ifndef MOEK_AD
#define MOEK_AD 2
#endif
#ifndef MOEK_WD
#define MOEK_WD 4
#endif
constexpr int AD = MOEK_AD, AST = AD + 1;  // activation ring
constexpr int WD = MOEK_WD, WST = WD + 1;  // weight ring slots
constexpr int A_STAGE = BM * BK * 2;       // 12 KB bf16
constexpr int B_STAGE = BK * BN / 2;       // 8 KB e2m1
constexpr int S_STAGE = (BK / 16) * BN;    // 1 KB plain scales
constexpr int W_STAGE = B_STAGE + S_STAGE;
constexpr int KG = 8, SPB = KG / 4;        // packed: 16-k groups per run, stages per run
constexpr int RUN_BYTES = 16 + KG * 64;    // 528: 16 bases (one per 8 positions) + KG rows of 4-bit offsets
constexpr int BLK_BYTES = 2 * RUN_BYTES;   // an item's two runs
constexpr int LAG = 2;                     // producer decodes stage s - LAG after issuing stage s
static_assert(LAG < 2 * SPB, "a staging slot must be decoded before it is refilled");
constexpr int A_OFF = 0, W_OFF = AST * A_STAGE, P_OFF = W_OFF + WST * W_STAGE, ROW_OFF = P_OFF + 2 * BLK_BYTES;
#ifndef MOEK_NU
#define MOEK_NU 4
#endif
constexpr int NU = MOEK_NU;  // unit slots: how many units the producer may publish ahead of the consumers
constexpr int ITEM_OFF = ROW_OFF + NU * BM * 4;  // NU row-table slots, then NU unit descriptors
constexpr int FLAG_OFF = ITEM_OFF + NU * 8 * 4, BAR_OFF = FLAG_OFF + 16;
constexpr int NBARS = 3 * WST + 2 + 2 * NU;  // full, empty, decoded per slot; 2 staging; NU unit-full; NU unit-empty
constexpr int SMEM_BYTES = BAR_OFF + NBARS * 8;


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
               "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(parity));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, %0;\n" ::"n"(CTHREADS)); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Marlin's dequant<nv_bfloat162, kFE2M1f, skip_flop=true>: out[1] from codes 3 and 7, out[0] from codes 2 and 6.
__device__ __forceinline__ void dq_fp4(uint32_t q, uint32_t* out) {
  out[1] = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
  q <<= 4;
  out[0] = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}
// Marlin's dequant_fp8_scales<nv_bfloat162, kFE4M3fn>: out[0] = (byte 0, byte 2), out[1] = (byte 1, byte 3).
__device__ __forceinline__ void dq_scales4(uint32_t q, uint32_t* out) {
  out[1] = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  out[0] = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
}
__device__ __forceinline__ uint32_t bcast_lo(uint32_t x) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %1, 0x1010;\n" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t bcast_hi(uint32_t x) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %1, 0x3232;\n" : "=r"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// vLLM's silu_and_mul for bf16: bf16(silu(float(g))) * u, rounded to bf16
__device__ __forceinline__ __nv_bfloat16 silu_mul(__nv_bfloat16 g, __nv_bfloat16 u) {
  float x = __bfloat162float(g);
  __nv_bfloat16 s = __float2bfloat16(x / (1.0f + expf(-x)));
  return __float2bfloat16(__bfloat162float(s) * __bfloat162float(u));
}



#ifdef MOEK_TRACE
// CTA 0 and 1, first 64 units: [0] consumer got the unit, [1] first weights ready, [2] last stage done,
// [3] epilogue done, [4] producer published it, [5] producer issued its last stage
__device__ long long g_tr[2][64][6];
#define TR(c, u, k, v) \
  if (blockIdx.x < 2 && (u) < 64) g_tr[blockIdx.x][u][k] = (v)
#else
#define TR(c, u, k, v)
#endif
struct Args {
  const __nv_bfloat16* A;
  const uint32_t* W;
  const uint8_t* S;  // plain scales (null when packed)
  const float* G;
  const int32_t* sorted_ids;
  const int32_t* expert_ids;
  const int32_t* num_post_pad;
  const float* topk_w;
  __nv_bfloat16* out;
  int num_ids, N, K;
  const uint8_t* SP;       // packed scales: per (expert, run of KG k-groups, 128 columns) RUN_BYTES
  const int32_t* exc_off;  // [E * K/16/KG + 1]
  const uint32_t* exc_ent;  // per exception scale: column (bits 0-15) | k-group in the run (16-23) | value (24-31)
  float* ws;               // stream-k partials: 2 slots per CTA (unit at its range start / end) x BM x 256
  int* cnt;                // per item: units done (self-resetting); cnt[CTR] / cnt[CTR + 1]: dynamic schedule
  int mode;                // 0: whole items handed out in order; 1: stream-k ranges; 2: split (below)
  // mode 2 (small batches): every item is nsplit units of K/nsplit, units round-robin in (block, column tile, k)
  // order so all SMs stream one expert at a time (Marlin's locality); units store fp32 partials
  // part[unit k][id][N] and moe_split_epilogue adds them and applies the epilogue
  int nsplit;
  float* part;
  int order;  // mode 2 unit order inside a block: 0 = (column tile, k), 1 = (k, column tile): SMs sweep the same rows
  // tiled layout (kcuda.tile_layout): W as [E][N/256][K/64][8 KB] (a stage of an item contiguous, in the shared
  // memory order) and SP as [E][N/256][K/16/KG][BLK_BYTES] (an item's run pair contiguous)
  int tiled;
};
constexpr int CTR = (1 << 16) - 2;  // dynamic-schedule counters at the end of cnt

struct Bars {
  uint64_t *full, *empty, *dec, *sbar, *ufull, *uempty;
};
__device__ __forceinline__ Bars bars_of(uint8_t* smem) {
  uint64_t* b = reinterpret_cast<uint64_t*>(smem + BAR_OFF);
  return {b, b + WST, b + 2 * WST, b + 3 * WST, b + 3 * WST + 2, b + 3 * WST + 2 + NU};
}

// Marlin 64-col tile index of an item's first column of each 128-column run
template <bool GATE_UP>
__device__ __forceinline__ void item_tiles(int N, int nt, int& t0, int& t1) {
  t0 = GATE_UP ? nt * 2 : nt * 4;
  t1 = GATE_UP ? N / 128 + nt * 2 : nt * 4 + 2;
}

// Stream-k: all items' k tiles in one sequence (position = item * ktiles + kt, item = block * NT + column tile);
// CTA c owns [start(c), start(c + 1)), cut at whole packed runs. Items cut by a range end are finished by whichever
// of their CTAs comes last (fp32 partials summed in CTA order: deterministic).
__device__ __forceinline__ int64_t range_start(int c, int64_t total, int grid) {
  return c >= grid ? total : (((int64_t)c * total / grid) & ~(int64_t)(SPB - 1));
}
__device__ __forceinline__ int owner(int64_t pos, int64_t total, int grid) {
  int c = (int)(pos * grid / total);
  if (c >= grid) c = grid - 1;
  while (c > 0 && range_start(c, total, grid) > pos) c--;
  while (c + 1 < grid && range_start(c + 1, total, grid) <= pos) c++;
  return c;
}
__device__ __forceinline__ int64_t total_positions(const Args& p) {
  return (int64_t)(p.num_post_pad[0] / BM) * (p.N / BN) * (p.K / BK);
}

struct Pend {
  int ws, ss, j, r0, r1, eo0, eo1, par, v;
  uint32_t pent;  // this lane's exception entry eo0 + lane (prefetched when the run was issued), if any
};

template <bool GATE_UP, bool PACKED>
__device__ __forceinline__ void producer(const Args& p, uint8_t* smem, const Bars& B) {
  const int lane = threadIdx.x & 31;
  const int N = p.N, K = p.K, NT = N / BN, ktiles = K / BK, n2 = N * 2, nb = K / 16 / KG;
  const int64_t total = total_positions(p);
  const int64_t P1 = range_start(blockIdx.x + 1, total, gridDim.x);
  int* units = reinterpret_cast<int*>(smem + ITEM_OFF);
  int* rows = reinterpret_cast<int*>(smem + ROW_OFF);
  uint8_t* stg = smem + P_OFF;
  int it = 0;
  uint32_t g = 0, gb = 0;
  Pend q0{}, q1{};

  auto decode = [&](const Pend& d) {
    if (d.j == 0) mbar_wait(B.sbar + d.ss, d.par);
    const uint8_t* blk = stg + d.ss * BLK_BYTES + (lane >> 4) * RUN_BYTES;
    uint8_t* plain = smem + W_OFF + d.ws * W_STAGE + B_STAGE;
    const uint32_t b4 = blk[lane & 15] * 0x01010101u;  // this lane's 8 positions share one base
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t q = *reinterpret_cast<const uint32_t*>(blk + 16 + (d.j * 4 + kk) * 64 + (lane & 15) * 4);
      uint2 v;
      v.x = (q & 0x0F0F0F0Fu) + b4;  // never carries: base + offset is the original byte (exceptions: offset 0)
      v.y = ((q >> 4) & 0x0F0F0F0Fu) + b4;
      *reinterpret_cast<uint2*>(plain + kk * 256 + lane * 8) = v;
    }
    __syncwarp();
    for (int i = d.eo0 + lane; i < d.eo1; i += 32) {
      const uint32_t en = i < d.eo0 + 32 ? d.pent : p.exc_ent[i];
      const int col = en & 0xFFFF, kg = (en >> 16) & 0xFF;
      if ((kg >> 2) != d.j) continue;  // another stage of the run
      int pos = col - d.r0 * 128;
      if (pos < 0 || pos >= 128) pos = (col - d.r1 * 128 >= 0 && col - d.r1 * 128 < 128) ? 128 + col - d.r1 * 128 : -1;
      if (pos >= 0) plain[(kg & 3) * 256 + pos] = en >> 24;
    }
    mbar_arrive(B.dec + d.ws);  // count 32: every lane's writes are released
  };

  const int n_items = (int)(total / ktiles);
  int64_t usplit = blockIdx.x;  // mode 2: next unit
  int cmb = -1, ce = -1, cnv = 0, crow[BM / 32];  // last block's expert, valid count and row ids (lane's share)
  for (int64_t pos = range_start(blockIdx.x, total, gridDim.x);;) {
    bool done;
    int item = 0, kt0 = 0, len = 0, mb = 0, nt = 0, e = -1, nvalid = 0, ks = 0;
    if (p.mode == 2) {
      done = usplit >= (int64_t)n_items * p.nsplit;
      if (!done) {
        const int per = NT * p.nsplit;
        const int blk = (int)(usplit / per), r = (int)(usplit - (int64_t)blk * per);
        const int ntl = p.order ? r % NT : r / p.nsplit;
        ks = p.order ? r / NT : r % p.nsplit;
        item = blk * NT + ntl;
        len = ktiles / p.nsplit;
        kt0 = ks * len;
      }
      usplit += gridDim.x;
    } else if (p.mode == 1) {
      done = pos >= P1;
      if (!done) {
        item = (int)(pos / ktiles);
        kt0 = (int)(pos - (int64_t)item * ktiles);
        len = (int)std::min<int64_t>(ktiles - kt0, P1 - pos);
      }
    } else {
      int nx = 0;
      if (lane == 0) nx = atomicAdd(p.cnt + CTR, 1);
      item = __shfl_sync(0xffffffffu, nx, 0);
      done = item >= n_items;
      len = done ? 0 : ktiles;  // len 0 = the consumers' stop signal
      if (done && lane == 0 && atomicAdd(p.cnt + CTR + 1, 1) == (int)gridDim.x - 1) {
        p.cnt[CTR] = 0;  // the last CTA out resets the schedule for the next launch
        p.cnt[CTR + 1] = 0;
      }
    }
    if (!done) {
      mb = item / NT;
      nt = item - mb * NT;
      if (mb != cmb) {  // consecutive units mostly share a block: load its expert and rows once
        cmb = mb;
        ce = p.expert_ids[mb];
        cnv = 0;
#pragma unroll
        for (int r0 = 0; r0 < BM / 32; r0++) {
          const int id = p.sorted_ids[(size_t)mb * BM + r0 * 32 + lane];
          const bool ok = id < p.num_ids;
          crow[r0] = ok ? id : -1;
          cnv += __popc(__ballot_sync(0xffffffffu, ok));  // valid rows are a prefix of the block
        }
      }
      e = ce;
      nvalid = cnv;
    }
    const int is = it % NU;
    if (it >= NU) mbar_wait(B.uempty + is, ((it / NU) - 1) & 1);
    if (!done) {
      int* rs = rows + is * BM;
#pragma unroll
      for (int r0 = 0; r0 < BM / 32; r0++) rs[r0 * 32 + lane] = crow[r0];
    }
    if (lane == 0) {
      int* u = units + is * 8;
      u[0] = e; u[1] = mb; u[2] = nt; u[3] = kt0; u[4] = len; u[5] = item; u[6] = nvalid; u[7] = ks;
    }
    __syncwarp();
    mbar_arrive(B.ufull + is);  // count 32
    if (lane == 0) TR(0, it, 4, clock64());
    it++;
    if (done) break;
    int t0, t1;
    item_tiles<GATE_UP>(N, nt, t0, t1);
    const int r0 = t0 / 2, r1 = t1 / 2;
    const uint32_t* We = p.W + (size_t)e * (K / 16) * n2;
    int eo0 = 0, eo1 = 0;
    uint32_t pent = 0;
    // exception lists are per (expert, item, run): this item's run bounds, one per lane (nb + 1 <= 32), loaded once
    const int myoff = (PACKED && lane <= nb) ? p.exc_off[((size_t)e * NT + nt) * nb + lane] : 0;
    for (int kt = kt0; kt < kt0 + len; kt++, g++) {
      const int ws = g % WST;
      if (lane == 0) {
        if (g >= WST) mbar_wait(B.empty + ws, ((g / WST) - 1) & 1);
        uint8_t* dst = smem + W_OFF + ws * W_STAGE;
        uint64_t* bar = B.full + ws;
        mbar_expect_tx(bar, PACKED ? B_STAGE : W_STAGE);
        if (p.tiled) {
          bulk_g2s(dst, reinterpret_cast<const uint8_t*>(p.W) + (((size_t)e * NT + nt) * ktiles + kt) * B_STAGE, B_STAGE, bar);
        } else
#pragma unroll
        for (int k16 = 0; k16 < 4; k16++) {
          const uint32_t* row = We + (size_t)(kt * 4 + k16) * n2;
          if (t1 == t0 + 2) {  // the two runs are adjacent (down): one 2 KB copy
            bulk_g2s(dst + k16 * 2048, row + t0 * 128, 2048, bar);
          } else {
            bulk_g2s(dst + k16 * 2048, row + t0 * 128, 1024, bar);
            bulk_g2s(dst + k16 * 2048 + 1024, row + t1 * 128, 1024, bar);
          }
          if (!PACKED) {
            const uint8_t* srow = p.S + ((size_t)e * (K / 16) + kt * 4 + k16) * N;
            bulk_g2s(dst + B_STAGE + k16 * 256, srow + t0 * 64, 128, bar);
            bulk_g2s(dst + B_STAGE + k16 * 256 + 128, srow + t1 * 64, 128, bar);
          }
        }
      }
      if (PACKED) {
        const int j = kt % SPB;
        if (j == 0) {
          const int b = kt / SPB, ss = gb & 1;
          if (lane == 0) {
            uint64_t* bar = B.sbar + ss;
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            mbar_expect_tx(bar, BLK_BYTES);
            const uint8_t* src = p.SP + ((size_t)e * nb + b) * (N / 128) * RUN_BYTES;
            if (p.tiled) {
              bulk_g2s(stg + ss * BLK_BYTES, p.SP + (((size_t)e * NT + nt) * nb + b) * BLK_BYTES, BLK_BYTES, bar);
            } else if (r1 == r0 + 1) {
              bulk_g2s(stg + ss * BLK_BYTES, src + (size_t)r0 * RUN_BYTES, BLK_BYTES, bar);
            } else {
              bulk_g2s(stg + ss * BLK_BYTES, src + (size_t)r0 * RUN_BYTES, RUN_BYTES, bar);
              bulk_g2s(stg + ss * BLK_BYTES + RUN_BYTES, src + (size_t)r1 * RUN_BYTES, RUN_BYTES, bar);
            }
          }
          eo0 = __shfl_sync(0xffffffffu, myoff, b);
          eo1 = __shfl_sync(0xffffffffu, myoff, b + 1);
          pent = eo0 + lane < eo1 ? p.exc_ent[eo0 + lane] : 0u;  // used LAG stages later: the load is not waited for
          gb++;
        }
        Pend d{ws, (int)((gb - 1) & 1), j, r0, r1, eo0, eo1, (int)(((gb - 1) >> 1) & 1), 1, pent};
        if (q1.v) {
          decode(q0);
          q0 = q1;
          q1 = d;
        } else if (q0.v) {
          q1 = d;
        } else {
          q0 = d;
        }
      }
      __syncwarp();
    }
    if (lane == 0) TR(0, it - 1, 5, clock64());
    pos += len;
  }
  if (PACKED) {
    if (q0.v) decode(q0);
    if (q1.v) decode(q1);
  }
}

// MMAs of one 16-k step for NFW row fragments from fragment f0, against the warp's 8 n8 column tiles
template <int NFW>
__device__ __forceinline__ void mma_rows(float (*acc)[8][4], const __nv_bfloat16* sa, int f0, int kk, int lane,
                                         const uint32_t (*bf)[2][2]) {
  uint32_t af[NFW][4];
#pragma unroll
  for (int mi = 0; mi < NFW; mi++) {
    int r = (f0 + mi) * 16 + (lane & 15);
    int ch = kk * 2 + (lane >> 4);
    ldmatrix_x4(af[mi], sa + r * BK + ((ch ^ (r & 7)) * 8));
  }
#pragma unroll
  for (int mi = 0; mi < NFW; mi++)
#pragma unroll
    for (int j = 0; j < 8; j++) mma_bf16(acc[mi][j], af[mi], bf[j >> 1][j & 1]);
}

// B fragments of one 16-k step: bf[group][n8 half][reg] for the warp's 4 n16 groups (gate_up: 2 gate + 2 up)
template <bool GATE_UP>
__device__ __forceinline__ void load_b(const uint8_t* sw, int kk, int slab, int lane, uint32_t (*bf)[2][2]) {
  uint32_t qs[4], sc[4];
  const uint8_t* plain = sw + B_STAGE + kk * 256;
  if (GATE_UP) {
    const int tg = slab >> 1, hp = slab & 1;  // gate tile tg of run 0, up tile tg of run 1; group pair hp
    const uint2 qg = *reinterpret_cast<const uint2*>(sw + (kk * 4 + tg) * 512 + lane * 16 + hp * 8);
    const uint2 qu = *reinterpret_cast<const uint2*>(sw + (kk * 4 + 2 + tg) * 512 + lane * 16 + hp * 8);
    qs[0] = qg.x; qs[1] = qg.y; qs[2] = qu.x; qs[3] = qu.y;
    const uint32_t sg = *reinterpret_cast<const uint32_t*>(plain + tg * 64 + (lane >> 2) * 8 + hp * 4);
    const uint32_t su = *reinterpret_cast<const uint32_t*>(plain + 128 + tg * 64 + (lane >> 2) * 8 + hp * 4);
    dq_scales4(sg, sc);
    dq_scales4(su, sc + 2);
  } else {
    const int4 q = *reinterpret_cast<const int4*>(sw + (kk * 4 + slab) * 512 + lane * 16);
    qs[0] = q.x; qs[1] = q.y; qs[2] = q.z; qs[3] = q.w;
    const uint2 s = *reinterpret_cast<const uint2*>(plain + slab * 64 + (lane >> 2) * 8);
    dq_scales4(s.x, sc);
    dq_scales4(s.y, sc + 2);
  }
#pragma unroll
  for (int g = 0; g < 4; g++) {
    uint32_t s0 = bcast_lo(sc[g]), s1 = bcast_hi(sc[g]), t[2];
    dq_fp4(qs[g] << 8, t);
    bf[g][0][0] = hmul2(t[0], s0);
    bf[g][0][1] = hmul2(t[1], s0);
    dq_fp4(qs[g], t);
    bf[g][1][0] = hmul2(t[0], s1);
    bf[g][1][1] = hmul2(t[1], s1);
  }
}


// one unit (len k tiles of an item from kt0) on the consumer warps; g0 = its first ring stage
template <int NF, int TOPK, bool GATE_UP, bool PACKED>
__device__ __forceinline__ void consume(const Args& p, uint8_t* smem, const Bars& B, const int* s_row, int e, int nt,
                                        uint32_t g0, int kt0, int len, int item, int ks, int uix) {
  constexpr int NFW0 = (NF + 1) / 2, NFW1 = NF / 2;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int slab = warp & 3, warp_m = warp >> 2;
  const int K = p.K, N = p.N, ktiles = K / BK;
  constexpr int A_CHUNKS = NF * 16 * (BK / 8);
  auto stage_a = [&](int s) { return reinterpret_cast<__nv_bfloat16*>(smem + A_OFF + s * A_STAGE); };
  auto load_a = [&](int kt) {  // kt: unit-local
    __nv_bfloat16* sa = stage_a(kt % AST);
#pragma unroll
    for (int i = 0; i < (A_CHUNKS + CTHREADS - 1) / CTHREADS; i++) {
      int idx = tid + i * CTHREADS;
      if (A_CHUNKS % CTHREADS == 0 || idx < A_CHUNKS) {
        int r = idx >> 3, ch = idx & 7;
        int id = s_row[r];
        bool ok = id >= 0;
        const __nv_bfloat16* src = p.A + (size_t)(ok ? id / TOPK : 0) * K + (kt0 + kt) * BK + ch * 8;
        cp_async16(sa + r * BK + ((ch ^ (r & 7)) * 8), src, ok);
      }
    }
  };

  float acc[NFW0][8][4];
#pragma unroll
  for (int i = 0; i < NFW0; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

  consumer_sync();  // everyone is done with the previous unit's activation buffers
#pragma unroll
  for (int k = 0; k < AD; k++) {
    if (k < len) load_a(k);
    cp_async_commit();
  }
  for (int kt = 0; kt < len; kt++) {
    const uint32_t g = g0 + kt;
    const int ws = g % WST;
    cp_async_wait<AD - 1>();
    consumer_sync();  // A[kt] visible to all consumers; all are done with A[kt-1]'s buffer
    if (kt + AD < len) load_a(kt + AD);
    cp_async_commit();
    mbar_wait(B.full + ws, (g / WST) & 1);
    if (PACKED) mbar_wait(B.dec + ws, (g / WST) & 1);
    if (kt == 0 && tid == 0) TR(0, uix, 1, clock64());
    const __nv_bfloat16* sa = stage_a(kt % AST);
    const uint8_t* sw = smem + W_OFF + ws * W_STAGE;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t bf[4][2][2];
      load_b<GATE_UP>(sw, kk, slab, lane, bf);
      if (warp_m == 0) {
        mma_rows<NFW0>(acc, sa, 0, kk, lane, bf);
      } else if (NFW1 > 0) {
        mma_rows<(NFW1 > 0 ? NFW1 : 1)>(acc, sa, NFW0, kk, lane, bf);
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(B.empty + ws);
  }
  cp_async_wait<0>();
  if (tid == 0) TR(0, uix, 2, clock64());

  const int nfw = warp_m == 0 ? NFW0 : NFW1, f0 = warp_m == 0 ? 0 : NFW0;
  if (p.mode == 2) {
    // fp32 partial rows part[ks][id][column] (gate_up: gate and up halves at their columns of the N rows of W)
    float* base = p.part + (size_t)ks * p.num_ids * N;
    int t0, t1;
    item_tiles<GATE_UP>(N, nt, t0, t1);
#pragma unroll
    for (int mi = 0; mi < NFW0; mi++) {
      if (mi >= nfw) break;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int r = (f0 + mi) * 16 + lane / 4 + half * 8;
        const int id = s_row[r];
        if (id < 0) continue;
        float* row = base + (size_t)id * N;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          int col;
          if (GATE_UP) {
            const int jg = j & 3;  // j < 4: gate, j >= 4: the matching up column
            col = (j < 4 ? 0 : N / 2) + nt * 128 + (slab >> 1) * 64 + (slab & 1) * 32 + (jg >> 1) * 16 + (jg & 1) * 8;
          } else {
            col = (t0 + slab) * 64 + j * 8;
          }
          *reinterpret_cast<float2*>(row + col + (lane & 3) * 2) = make_float2(acc[mi][j][half * 2], acc[mi][j][half * 2 + 1]);
        }
      }
    }
    if (tid == 0) TR(0, uix, 3, clock64());
    return;
  }
  if (len < ktiles) {
    // a piece of an item cut by stream-k: publish the partial sums; the item's last piece adds them in CTA order
    const int64_t total = total_positions(p);
    const int grid = gridDim.x, c = blockIdx.x;
    const int64_t ist = (int64_t)item * ktiles;
    const int slot_mine = 2 * c + (ist + kt0 == range_start(c, total, grid) ? 0 : 1);
    float* slot = p.ws + (size_t)slot_mine * BM * 256;
#pragma unroll
    for (int mi = 0; mi < NFW0; mi++) {
      if (mi >= nfw) break;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int r = (f0 + mi) * 16 + lane / 4 + half * 8;
        if (s_row[r] < 0) continue;
#pragma unroll
        for (int j = 0; j < 8; j++)
          __stcg(reinterpret_cast<float2*>(slot + r * 256 + slab * 64 + j * 8 + (lane & 3) * 2),
                 make_float2(acc[mi][j][half * 2], acc[mi][j][half * 2 + 1]));
      }
    }
    __threadfence();
    consumer_sync();
    const int c0 = owner(ist, total, grid), c1 = owner(ist + ktiles - 1, total, grid);
    volatile int* flag = reinterpret_cast<volatile int*>(smem + FLAG_OFF);
    if (tid == 0) *flag = atomicAdd(p.cnt + item, 1) == c1 - c0;
    consumer_sync();
    if (!*flag) return;
    __threadfence();
#pragma unroll
    for (int mi = 0; mi < NFW0; mi++) {
      if (mi >= nfw) break;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int r = (f0 + mi) * 16 + lane / 4 + half * 8;
        if (s_row[r] < 0) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          float x = 0.f, y = 0.f;
          for (int cc = c0; cc <= c1; cc++) {
            const int sl = 2 * cc + ((cc == c0 && range_start(cc, total, grid) < ist) ? 1 : 0);
            const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ws + (size_t)sl * BM * 256 + r * 256 +
                                                                    slab * 64 + j * 8 + (lane & 3) * 2));
            x += v.x;
            y += v.y;
          }
          acc[mi][j][half * 2] = x;
          acc[mi][j][half * 2 + 1] = y;
        }
      }
    }
    if (tid == 0) p.cnt[item] = 0;  // ready for the next launch
  }

  const float gs = p.G[e];
  int t0, t1;
  item_tiles<GATE_UP>(N, nt, t0, t1);
#pragma unroll
  for (int mi = 0; mi < NFW0; mi++) {
    if (mi >= nfw) break;
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int r = (f0 + mi) * 16 + lane / 4 + half * 8;
      const int id = s_row[r];
      if (id < 0) continue;
      if (GATE_UP) {
        __nv_bfloat16* orow = p.out + (size_t)id * (N / 2) + nt * 128 + (slab >> 1) * 64 + (slab & 1) * 32 + (lane & 3) * 2;
#pragma unroll
        for (int jg = 0; jg < 4; jg++) {  // gate group pair x n8 half; up partner at jg + 4
          __nv_bfloat162 gv = __floats2bfloat162_rn(acc[mi][jg][half * 2] * gs, acc[mi][jg][half * 2 + 1] * gs);
          __nv_bfloat162 uv =
              __floats2bfloat162_rn(acc[mi][jg + 4][half * 2] * gs, acc[mi][jg + 4][half * 2 + 1] * gs);
          __nv_bfloat162 o;
          o.x = silu_mul(gv.x, uv.x);
          o.y = silu_mul(gv.y, uv.y);
          *reinterpret_cast<__nv_bfloat162*>(orow + (jg >> 1) * 16 + (jg & 1) * 8) = o;
        }
      } else {
        float f = gs;
        f *= p.topk_w[id];
        __nv_bfloat16* orow = p.out + (size_t)id * N + (t0 + slab) * 64 + (lane & 3) * 2;
#pragma unroll
        for (int j = 0; j < 8; j++)
          *reinterpret_cast<__nv_bfloat162*>(orow + j * 8) =
              __floats2bfloat162_rn(acc[mi][j][half * 2] * f, acc[mi][j][half * 2 + 1] * f);
      }
    }
  }
}

// a unit with no valid rows (not expected): keep the rings moving
template <bool PACKED>
__device__ __forceinline__ void drain(const Bars& B, uint32_t g0, int len) {
  const int lane = threadIdx.x & 31;
  for (int kt = 0; kt < len; kt++) {
    const uint32_t g = g0 + kt;
    mbar_wait(B.full + g % WST, (g / WST) & 1);
    if (PACKED) mbar_wait(B.dec + g % WST, (g / WST) & 1);
    __syncwarp();
    if (lane == 0) mbar_arrive(B.empty + g % WST);
  }
}

template <int TOPK, bool GATE_UP, bool PACKED>
__global__ void __launch_bounds__(THREADS, 1) moe_w4a16_kernel(Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Bars B = bars_of(smem);
  const int tid = threadIdx.x, lane = tid & 31;
  if (tid == 0) {
    for (int b = 0; b < WST; b++) {
      mbar_init(B.full + b, 1);              // producer's expect_tx + the copies' bytes
      mbar_init(B.empty + b, CTHREADS / 32);  // one arrive per consumer warp
      mbar_init(B.dec + b, 32);              // every producer lane
    }
    for (int b = 0; b < 2; b++) mbar_init(B.sbar + b, 1);
    for (int b = 0; b < NU; b++) {
      mbar_init(B.ufull + b, 32);             // every producer lane (row table)
      mbar_init(B.uempty + b, CTHREADS / 32);  // one arrive per consumer warp
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  if (tid >= CTHREADS) {
    producer<GATE_UP, PACKED>(p, smem, B);
    return;
  }
  const int* units = reinterpret_cast<const int*>(smem + ITEM_OFF);
  const int* rows = reinterpret_cast<const int*>(smem + ROW_OFF);
  uint32_t g = 0;
  for (int it = 0;; it++) {
    const int is = it % NU;
    mbar_wait(B.ufull + is, (it / NU) & 1);
    const int* u = units + is * 8;
    const int e = u[0], nt = u[2], kt0 = u[3], len = u[4], item = u[5], nvalid = u[6], ks = u[7];
    if (len == 0) break;
    const int* s_row = rows + is * BM;
    if (tid == 0) TR(0, it, 0, clock64());
    switch ((nvalid + 15) / 16) {
      case 1: consume<1, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      case 2: consume<2, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      case 3: consume<3, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      case 4: consume<4, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      case 5: consume<5, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      case 6: consume<6, TOPK, GATE_UP, PACKED>(p, smem, B, s_row, e, nt, g, kt0, len, item, ks, it); break;
      default: drain<PACKED>(B, g, len); break;
    }
    g += len;
    __syncwarp();
    if (lane == 0) mbar_arrive(B.uempty + is);  // this warp is done with the unit's rows
  }
}

// gate_up: out = act [num_ids, N/2] (silu(gate) * up); down: out = [num_ids, N] rows times the router weight.
// Scales: plain Marlin S [E, K/16, N], or (SP non-empty) packed SP + exc_off + exc_ent (mbx_moek.pack_scales).
// ws: >= 2 * #SMs * BM * 256 fp32; cnt: >= items int32, zero (kept zero by the kernel).
void moe_gemm(torch::Tensor A, torch::Tensor W, torch::Tensor S, torch::Tensor G, torch::Tensor sorted_ids,
              torch::Tensor expert_ids, torch::Tensor num_post_pad, torch::Tensor topk_w, torch::Tensor out,
              int64_t top_k, bool gate_up, torch::Tensor SP, torch::Tensor exc_off, torch::Tensor exc_ent,
              torch::Tensor ws, torch::Tensor cnt, int64_t mode, int64_t nsplit, torch::Tensor part, int64_t order,
              bool tiled) {
  const int K = A.size(1);
  const int N = W.size(2) / 2;
  const int num_ids = out.size(0);
  const bool packed = SP.numel() > 0;
  if (num_ids == 0) return;  // empty batch (vLLM dummy runs)
  TORCH_CHECK(K % BK == 0 && N % BN == 0 && (K / BK) % SPB == 0, "K % 128 and N % 256 must be 0");
  TORCH_CHECK(W.size(1) == K / 16, "W must be Marlin [E, K/16, 2N] int32 (or its tiled permutation, same shape)");
  TORCH_CHECK(tiled == packed, "packed scales come with the tiled layout (exception lists are per item)");
  if (packed) {
    TORCH_CHECK((K / 16) % KG == 0, "packed scales: K/16 % KG");
    TORCH_CHECK(SP.numel() == W.size(0) * (int64_t)(K / 16 / KG) * (N / 128) * RUN_BYTES, "SP size");
    TORCH_CHECK(exc_ent.numel() == 0 || exc_ent.scalar_type() == torch::kInt32, "exception entries are int32");
  } else {
    TORCH_CHECK(S.size(1) == K / 16 && S.size(2) == N && S.element_size() == 1, "S must be [E, K/16, N] fp8");
  }
  TORCH_CHECK(out.size(1) == (gate_up ? N / 2 : N), "out width");
  TORCH_CHECK(!gate_up || (N / 2) % 128 == 0, "gate_up N/2 % 128");
  static int sms = 0;
  if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, A.get_device());
  const int64_t items = (sorted_ids.size(0) / BM) * (N / BN);
  // at most this many k-tile positions exist (blocks that can be non-empty: ids / BM + min(E, ids))
  const int64_t E = W.size(0);
  const int64_t maxpos =
      std::min<int64_t>(sorted_ids.size(0) / BM, (num_ids + BM - 1) / BM + std::min<int64_t>(E, num_ids)) *
      (N / BN) * (K / BK);
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode");
  if (mode == 2) {
    TORCH_CHECK(nsplit >= 1 && (K / BK) % nsplit == 0 && (K / BK / nsplit) % SPB == 0, "nsplit must cut K into whole runs");
    TORCH_CHECK(part.numel() >= nsplit * (int64_t)num_ids * N && part.scalar_type() == torch::kFloat32, "split partials");
  }
  const int64_t units = mode == 2 ? maxpos / (K / BK) * nsplit : mode == 1 ? maxpos / SPB : maxpos / (K / BK);
  const int grid = (int)std::max<int64_t>(1, std::min<int64_t>(sms, units));
  TORCH_CHECK(ws.numel() >= (int64_t)2 * grid * BM * 256 && ws.scalar_type() == torch::kFloat32, "stream-k workspace");
  TORCH_CHECK(cnt.numel() == (1 << 16) && items <= CTR && cnt.scalar_type() == torch::kInt32, "stream-k counters");
  auto stream = c10::cuda::getCurrentCUDAStream();
  Args p{reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
         reinterpret_cast<const uint32_t*>(W.data_ptr()),
         packed ? nullptr : reinterpret_cast<const uint8_t*>(S.data_ptr()),
         G.data_ptr<float>(),
         sorted_ids.data_ptr<int32_t>(),
         expert_ids.data_ptr<int32_t>(),
         num_post_pad.data_ptr<int32_t>(),
         gate_up ? nullptr : topk_w.data_ptr<float>(),
         reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
         num_ids,
         N,
         K,
         packed ? reinterpret_cast<const uint8_t*>(SP.data_ptr()) : nullptr,
         packed ? exc_off.data_ptr<int32_t>() : nullptr,
         packed && exc_ent.numel() ? reinterpret_cast<const uint32_t*>(exc_ent.data_ptr()) : nullptr,
         ws.data_ptr<float>(),
         cnt.data_ptr<int32_t>(),
         (int)mode,
         (int)nsplit,
         mode == 2 ? part.data_ptr<float>() : nullptr,
         (int)order,
         tiled ? 1 : 0};
  auto launch = [&](void (*kern)(Args)) {
    static void (*done[8])(Args) = {};  // kernels whose shared-memory limit is already raised
    int i = 0;
    while (i < 8 && done[i] && done[i] != kern) i++;
    if (i < 8 && done[i] != kern) {
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
      done[i] = kern;
    }
    kern<<<grid, THREADS, SMEM_BYTES, stream>>>(p);
  };
  if (gate_up && top_k == 5)
    packed ? launch(moe_w4a16_kernel<5, true, true>) : launch(moe_w4a16_kernel<5, true, false>);
  else if (!gate_up && top_k == 1)
    packed ? launch(moe_w4a16_kernel<1, false, true>) : launch(moe_w4a16_kernel<1, false, false>);
  else
    TORCH_CHECK(false, "unsupported top_k for this kernel");
}

// mode 2 epilogue: out from the nsplit fp32 partials, added in unit order (deterministic)
//   gate_up: act[id][c] = silu(bf16(g * gs)) * bf16(u * gs) (c < N/2; g, u = partial sums of columns c and N/2 + c)
//   down:    out[id][n] = bf16(sum * (gs * w[id]))
template <bool GATE_UP>
__global__ void moe_split_epilogue_kernel(const float* __restrict__ part, int nsplit, int num_ids, int N,
                                          const int32_t* __restrict__ topk_ids, const float* __restrict__ G,
                                          const float* __restrict__ topk_w, __nv_bfloat16* __restrict__ out) {
  const int NO = GATE_UP ? N / 2 : N;
  const int64_t i2 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // pair of output columns
  if (i2 >= (int64_t)num_ids * NO / 2) return;
  const int id = (int)(i2 / (NO / 2)), c = (int)(i2 - (int64_t)id * (NO / 2)) * 2;
  const float gs = G[topk_ids[id]];
  const size_t stride = (size_t)num_ids * N;
  const float* r = part + (size_t)id * N;
  float2 a = make_float2(0.f, 0.f), b = make_float2(0.f, 0.f);
  for (int s = 0; s < nsplit; s++) {
    const float2 v = *reinterpret_cast<const float2*>(r + s * stride + c);
    a.x += v.x;
    a.y += v.y;
    if (GATE_UP) {
      const float2 w = *reinterpret_cast<const float2*>(r + s * stride + N / 2 + c);
      b.x += w.x;
      b.y += w.y;
    }
  }
  __nv_bfloat162 o;
  if (GATE_UP) {
    const __nv_bfloat162 gv = __floats2bfloat162_rn(a.x * gs, a.y * gs), uv = __floats2bfloat162_rn(b.x * gs, b.y * gs);
    o.x = silu_mul(gv.x, uv.x);
    o.y = silu_mul(gv.y, uv.y);
  } else {
    float f = gs;
    f *= topk_w[id];
    o = __floats2bfloat162_rn(a.x * f, a.y * f);
  }
  *reinterpret_cast<__nv_bfloat162*>(out + (size_t)id * NO + c) = o;
}

void moe_split_epilogue(torch::Tensor part, int64_t nsplit, torch::Tensor topk_ids, torch::Tensor G,
                        torch::Tensor topk_w, torch::Tensor out, bool gate_up, int64_t N) {
  const int num_ids = out.size(0);
  const int64_t pairs = (int64_t)num_ids * out.size(1) / 2;
  const int threads = 256;
  const int blocks = (int)((pairs + threads - 1) / threads);
  if (blocks == 0) return;  // empty batch
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (gate_up)
    moe_split_epilogue_kernel<true><<<blocks, threads, 0, stream>>>(part.data_ptr<float>(), (int)nsplit, num_ids, (int)N,
                                                                    topk_ids.data_ptr<int32_t>(), G.data_ptr<float>(),
                                                                    nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  else
    moe_split_epilogue_kernel<false><<<blocks, threads, 0, stream>>>(
        part.data_ptr<float>(), (int)nsplit, num_ids, (int)N, topk_ids.data_ptr<int32_t>(), G.data_ptr<float>(),
        topk_w.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
}

}  // namespace moek

#ifdef MOEK_TRACE
torch::Tensor trace() {
  auto t = torch::empty({2, 64, 6}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), moek::g_tr, sizeof(moek::g_tr));
  return t;
}
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOEK_TRACE
  m.def("trace", &trace);
#endif
  m.def("moe_gemm", &moek::moe_gemm);
  m.def("moe_split_epilogue", &moek::moe_split_epilogue);
}
