#include "merc-tq.cuh"

// Row layout (bytes): [pos i32 | inv_r f32 x G | latent norm f16 | rope norm f16 | rope codes rd/2 | latent codes ceil(r/2)]
// Codes: index = #{boundaries < x / n} (torch.bucketize), n = fp16(||x||) clamped at 1e-12; decode centroid[index] * n.
// Same arithmetic as the CPU reference (ggml-cpu/ops.cpp) and mercurius.models.qat.tq_roundtrip.

#define MERC_TQ_THREADS 256

static __device__ __forceinline__ uint8_t merc_tq_index(float u, const float * bnd) {
    uint8_t i = 0;
#pragma unroll
    for (int j = 0; j < 15; ++j) {
        i += bnd[j] < u;
    }
    return i;
}

static __device__ float merc_block_sum(float v, float * shared) {
    v = warp_reduce_sum(v);
    const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < blockDim.x / WARP_SIZE ? shared[threadIdx.x] : 0.0f;
    v = warp_reduce_sum(v);                      // the total is in warp 0 only: broadcast it
    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();
    v = shared[0];
    __syncthreads();
    return v;
}

// encode n values of x into packed nibble codes; returns the fp16 norm (all threads)
static __device__ half merc_tq_encode(const float * x, int n, const float * cb, uint8_t * codes, float * shared) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = merc_block_sum(ss, shared);
    const half nh = __float2half(sqrtf(ss));
    float nf = __half2float(nh);
    nf = nf > 1e-12f ? nf : 1e-12f;
    const float * bnd = cb + 16;
    for (int p = threadIdx.x; p < (n + 1) / 2; p += blockDim.x) {
        const int i = 2 * p;
        const uint8_t lo = merc_tq_index(x[i] / nf, bnd);
        const uint8_t hi = i + 1 < n ? merc_tq_index(x[i + 1] / nf, bnd) : 0;
        codes[p] = lo | (uint8_t) (hi << 4);
    }
    return nh;
}

static __global__ void merc_tq_pack_kernel(
        const char * c, const char * kr, const char * ir, const int32_t * pos, const float * cbl, const float * cbr,
        char * dst, int64_t nbc1, int64_t nbk1, int64_t nbi1, int64_t nbd1, int r, int rd, int G, int words) {
    __shared__ float shared[32];
    const int64_t t = blockIdx.x;
    uint8_t * row = (uint8_t *) (dst + t * nbd1);
    for (int i = threadIdx.x; i < 4 * words; i += blockDim.x) {
        row[i] = 0;
    }
    __syncthreads();
    uint8_t * rope_codes = row + 4 + 4 * G + 4;
    uint8_t * lat_codes  = rope_codes + rd / 2;
    const half nr = merc_tq_encode((const float *) (kr + t * nbk1), rd, cbr, rope_codes, shared);
    const half nl = merc_tq_encode((const float *) (c  + t * nbc1), r,  cbl, lat_codes,  shared);
    if (threadIdx.x == 0) {
        memcpy(row, pos + t, 4);
        memcpy(row + 4, ir + t * nbi1, 4 * G);
        memcpy(row + 4 + 4 * G,     &nl, 2);
        memcpy(row + 4 + 4 * G + 2, &nr, 2);
    }
}

static __global__ void merc_tq_unpack_kernel(
        const char * pk, const float * cbl, const float * cbr, char * dst, int64_t ne1, int64_t ne2,
        int64_t nb1, int64_t nb2, int64_t nb3, int64_t nbd1, int r, int rd, int G, int mode) {
    const int64_t t  = blockIdx.x;
    const int64_t i1 = t % ne1, i2 = (t / ne1) % ne2, i3 = t / (ne1 * ne2);
    const uint8_t * row = (const uint8_t *) (pk + i1 * nb1 + i2 * nb2 + i3 * nb3);
    if (mode == 1) {
        if (threadIdx.x == 0) {
            memcpy((int32_t *) dst + t, row, 4);
        }
        return;
    }
    float * y = (float *) (dst + t * nbd1);
    half nlh, nrh;
    memcpy(&nlh, row + 4 + 4 * G, 2);
    memcpy(&nrh, row + 4 + 4 * G + 2, 2);
    float nl = __half2float(nlh), nr = __half2float(nrh);
    nl = nl > 1e-12f ? nl : 1e-12f;
    nr = nr > 1e-12f ? nr : 1e-12f;
    const uint8_t * rope_codes = row + 4 + 4 * G + 4;
    const uint8_t * lat_codes  = rope_codes + rd / 2;
    for (int i = threadIdx.x; i < r + rd + G; i += blockDim.x) {
        if (i < r) {
            const uint8_t b = lat_codes[i / 2];
            y[i] = cbl[(i & 1) ? (b >> 4) : (b & 15)] * nl;
        } else if (i < r + rd) {
            const int j = i - r;
            const uint8_t b = rope_codes[j / 2];
            y[i] = cbr[(j & 1) ? (b >> 4) : (b & 15)] * nr;
        } else {
            float v;
            memcpy(&v, row + 4 + 4 * (i - r - rd), 4);
            y[i] = v;
        }
    }
}

void ggml_cuda_op_merc_tq_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * c = dst->src[0], * kr = dst->src[1], * ir = dst->src[2], * pos = dst->src[3];
    GGML_ASSERT(ggml_is_contiguous(pos));
    const int words = ggml_get_op_params_i32(dst, 0), r = ggml_get_op_params_i32(dst, 1);
    const int rd = ggml_get_op_params_i32(dst, 2), G = ggml_get_op_params_i32(dst, 3);
    const int64_t T = dst->ne[1];
    if (T == 0) {
        return;
    }
    merc_tq_pack_kernel<<<T, MERC_TQ_THREADS, 0, ctx.stream()>>>(
        (const char *) c->data, (const char *) kr->data, (const char *) ir->data, (const int32_t *) pos->data,
        (const float *) dst->src[4]->data, (const float *) dst->src[5]->data, (char *) dst->data,
        c->nb[1], kr->nb[1], ir->nb[1], dst->nb[1], r, rd, G, words);
}

void ggml_cuda_op_merc_tq_unpack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pk = dst->src[0];
    const int mode = ggml_get_op_params_i32(dst, 0), r = ggml_get_op_params_i32(dst, 1);
    const int rd = ggml_get_op_params_i32(dst, 2), G = ggml_get_op_params_i32(dst, 3);
    const int64_t n = ggml_nrows(pk);
    if (n == 0) {
        return;
    }
    merc_tq_unpack_kernel<<<n, MERC_TQ_THREADS, 0, ctx.stream()>>>(
        (const char *) pk->data, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
        (char *) dst->data, pk->ne[1], pk->ne[2], pk->nb[1], pk->nb[2], pk->nb[3], dst->nb[1], r, rd, G, mode);
}

// ---------------------------------------------------------------------------------------------------------------
// GGML_OP_MERC_TQ_ATTN: absorbed-MLA attention straight from the packed cache (flash-decoding, split over cells).
// Block = (chunk of 64 cells, query token). In the absorbed form all H (= 16) heads share one key per cell, so a
// chunk is one tensor-core problem: decode the chunk into an f16 tile K [64 x E] (codebook * norm; the 64-dim RoPE
// key un-rotated by R0^T and roped), S = Q [16 x E] K^T (WMMA, f32 accumulate), per-group 1/rms, scale, mask,
// chunk softmax, O = P [16 x 64] C [64 x r] (C = the latent part of the same tile). The chunk's (max, sum, O) go to
// a workspace that merc_tq_attn_reduce combines across chunks.

#include <mma.h>

#define MERC_ATTN_CHUNK   64
#define MERC_ATTN_THREADS 512
#define MERC_ATTN_H       16                     // query heads: one 16-row MMA tile

static __host__ __device__ inline int merc_pad16(int x) { return (x + 15) / 16 * 16; }

template <typename mask_t>
static __global__ void __launch_bounds__(MERC_ATTN_THREADS) merc_tq_attn_chunk(
        const float * q, int64_t q_nb1, int64_t q_nb2, const char * pk, int64_t nb_cell, int row_words,
        const float * cbl_g, const float * cbr_g, const void * U, bool U_f16, const mask_t * mask, int64_t mask_nb1,
        float * part_o, float * part_ml, int n_kv, int n_chunks, int G, int r, int rd, float scale, float theta_scale) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    const int E = rd + r, Ep = merc_pad16(E), rp = merc_pad16(r);
    half  * Ks  = (half *) smem_raw;                          // [CHUNK][Ep]   decoded keys (latent part = C)
    half  * Qs  = Ks + MERC_ATTN_CHUNK * Ep;                  // [16][Ep]
    float * Us  = (float *) (Qs + MERC_ATTN_H * Ep);          // [rd][rd] transposed: Us[k*rd + m] = U[m][k]
    float * Ss  = Us + rd * rd;                               // [16][CHUNK] scores
    half  * Ps  = (half *) (Ss + MERC_ATTN_H * MERC_ATTN_CHUNK);   // [16][CHUNK] probabilities
    float * raw = (float *) (Ps + MERC_ATTN_H * MERC_ATTN_CHUNK);  // [warps][rd]
    float * cb  = raw + (MERC_ATTN_THREADS / WARP_SIZE) * rd;      // [32]: latent, rope centroids
    uint32_t * rows = (uint32_t *) (cb + 32);                      // [CHUNK][row_words] packed cells, on chip
    const int t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int nwarps = MERC_ATTN_THREADS / WARP_SIZE;
    const int rep = MERC_ATTN_H / G;

    for (int i = tid; i < MERC_ATTN_H * Ep; i += blockDim.x) {
        const int h = i / Ep, e = i % Ep;
        Qs[i] = __float2half(e < E ? *(const float *) ((const char *) q + t * q_nb1 + h * q_nb2 + e * sizeof(float)) : 0.0f);
    }
    for (int i = tid; i < rd * rd; i += blockDim.x) {
        const int m = i / rd, k = i % rd;
        Us[k * rd + m] = U_f16 ? __half2float(((const half *) U)[i]) : ((const float *) U)[i];
    }
    if (tid < 16) { cb[tid] = cbl_g[tid]; cb[16 + tid] = cbr_g[tid]; }

    // a fixed grid of blocks walks the chunks: Q, R0^T and the codebooks are loaded once per block
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gridDim.x) {
    const int j0 = chunk * MERC_ATTN_CHUNK;
    const int nc = min(MERC_ATTN_CHUNK, n_kv - j0);
    __syncthreads();                                   // previous chunk's tile fully consumed
    for (int i = tid; i < MERC_ATTN_CHUNK * row_words; i += blockDim.x) {   // coalesced 4-byte loads
        const int jl = i / row_words, w = i % row_words;
        rows[i] = jl < nc ? ((const uint32_t *) (pk + (int64_t) (j0 + jl) * nb_cell))[w] : 0u;
    }
    __syncthreads();

    // decode the chunk: latent part (all threads, coalesced code reads), zero padding and empty cells
    for (int i = tid; i < MERC_ATTN_CHUNK * (Ep - rd); i += blockDim.x) {
        const int jl = i / (Ep - rd), e = i % (Ep - rd);
        float v = 0.0f;
        if (jl < nc && e < r) {
            const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
            half nlh; memcpy(&nlh, row + 4 + 4 * G, 2);
            float nl = __half2float(nlh); nl = nl > 1e-12f ? nl : 1e-12f;
            const uint8_t b = row[4 + 4 * G + 4 + rd / 2 + e / 2];
            v = cb[(e & 1) ? (b >> 4) : (b & 15)] * nl;
        }
        Ks[jl * Ep + rd + e] = __float2half(v);
    }
    // RoPE key: one warp per cell; lane i owns the NeoX pair (i, i + rd/2)
    for (int jl = warp; jl < MERC_ATTN_CHUNK; jl += nwarps) {
        if (jl >= nc) {
            for (int k = lane; k < rd; k += WARP_SIZE) Ks[jl * Ep + k] = __float2half(0.0f);
            continue;
        }
        const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
        int32_t pos; memcpy(&pos, row, 4);
        half nrh; memcpy(&nrh, row + 4 + 4 * G + 2, 2);
        float nr = __half2float(nrh); nr = nr > 1e-12f ? nr : 1e-12f;
        float * rw = raw + warp * rd;
        for (int k = lane; k < rd; k += WARP_SIZE) {
            const uint8_t b = row[4 + 4 * G + 4 + k / 2];
            rw[k] = cb[16 + ((k & 1) ? (b >> 4) : (b & 15))] * nr;
        }
        __syncwarp();
        for (int i = lane; i < rd / 2; i += WARP_SIZE) {
            float x0 = 0.0f, x1 = 0.0f;
            for (int k = 0; k < rd; ++k) {
                x0 += Us[k * rd + i] * rw[k];
                x1 += Us[k * rd + i + rd / 2] * rw[k];
            }
            const float th = (float) pos * powf(theta_scale, (float) i);
            const float cs = cosf(th), sn = sinf(th);
            Ks[jl * Ep + i]          = __float2half(x0 * cs - x1 * sn);
            Ks[jl * Ep + i + rd / 2] = __float2half(x0 * sn + x1 * cs);
        }
        __syncwarp();
    }
    __syncthreads();

    // S = Q K^T: 4 column tiles of 16 cells, one warp each, over Ep / 16 steps
    if (warp < MERC_ATTN_CHUNK / 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int e0 = 0; e0 < Ep; e0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
            wmma::load_matrix_sync(fa, Qs + e0, Ep);
            wmma::load_matrix_sync(fb, Ks + warp * 16 * Ep + e0, Ep);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(Ss + warp * 16, acc, MERC_ATTN_CHUNK, wmma::mem_row_major);
    }
    __syncthreads();

    // per-group 1/rms, scale, mask; chunk softmax per head (warp per head)
    for (int i = tid; i < MERC_ATTN_H * MERC_ATTN_CHUNK; i += blockDim.x) {
        const int h = i / MERC_ATTN_CHUNK, jl = i % MERC_ATTN_CHUNK;
        float s = -INFINITY;
        if (jl < nc) {
            const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
            float rms; memcpy(&rms, row + 4 + 4 * (h / rep), 4);
            const float m = mask ? (float) mask[t * mask_nb1 + j0 + jl] : 0.0f;
            s = Ss[i] / (rms > 1e-6f ? rms : 1e-6f) * scale + m;
        }
        Ss[i] = s;
    }
    __syncthreads();
    float * ml = part_ml + ((int64_t) t * n_chunks + chunk) * 2 * MERC_ATTN_H;
    for (int h = warp; h < MERC_ATTN_H; h += nwarps) {
        float mx = -INFINITY;
        for (int jl = lane; jl < MERC_ATTN_CHUNK; jl += WARP_SIZE) mx = fmaxf(mx, Ss[h * MERC_ATTN_CHUNK + jl]);
        mx = warp_reduce_max(mx);
        float l = 0.0f;
        for (int jl = lane; jl < MERC_ATTN_CHUNK; jl += WARP_SIZE) {
            const float s = Ss[h * MERC_ATTN_CHUNK + jl];
            const float p = mx == -INFINITY ? 0.0f : expf(s - mx);
            Ps[h * MERC_ATTN_CHUNK + jl] = __float2half(p);
            l += p;
        }
        l = warp_reduce_sum(l);
        if (lane == 0) { ml[h] = mx; ml[MERC_ATTN_H + h] = l; }
    }
    __syncthreads();

    // O = P C: rp / 16 column tiles over the warps, K dim = 64 cells; straight to the workspace
    float * o = part_o + ((int64_t) t * n_chunks + chunk) * MERC_ATTN_H * rp;
    for (int n0 = warp * 16; n0 < rp; n0 += nwarps * 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int k0 = 0; k0 < MERC_ATTN_CHUNK; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
            wmma::load_matrix_sync(fa, Ps + k0, MERC_ATTN_CHUNK);
            wmma::load_matrix_sync(fb, Ks + k0 * Ep + rd + n0, Ep);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(o + n0, acc, rp, wmma::mem_row_major);
    }
    }                                                  // chunk loop
}

static __global__ void merc_tq_attn_reduce(const float * part_o, const float * part_ml, float * dst, int64_t d_nb1,
                                           int64_t d_nb2, int n_chunks, int r, int rp) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float * ml = part_ml + (int64_t) t * n_chunks * 2 * MERC_ATTN_H;
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, ml[c * 2 * MERC_ATTN_H + h]);
    float L = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = ml[c * 2 * MERC_ATTN_H + h];
        if (m != -INFINITY) L += ml[c * 2 * MERC_ATTN_H + MERC_ATTN_H + h] * expf(m - M);
    }
    float * y = (float *) ((char *) dst + t * d_nb1 + h * d_nb2);
    for (int e = threadIdx.x; e < r; e += blockDim.x) {
        float v = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const float m = ml[c * 2 * MERC_ATTN_H + h];
            if (m != -INFINITY) v += part_o[(((int64_t) t * n_chunks + c) * MERC_ATTN_H + h) * rp + e] * expf(m - M);
        }
        y[e] = L > 0.0f ? v / L : 0.0f;
    }
}

// ---- prefill (many query tokens): decode the cache once to f16, then flash attention per query token ----------
// Kd [n_kv][Ep] f16: [rope_neox(R0^T k_rope, pos) | c | 0 pad]. One block per 64 cells.
static __global__ void __launch_bounds__(256) merc_tq_decode_f16(
        const char * pk, int64_t nb_cell, int row_words, const float * cbl_g, const float * cbr_g, const void * U,
        bool U_f16, half * Kd, int Ep, int n_kv, int G, int r, int rd, float theta_scale, float * rmsd) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float * Us  = (float *) smem_raw;                         // [rd][rd] transposed
    float * raw = Us + rd * rd;                               // [8 warps][rd]
    float * cb  = raw + 8 * rd;                               // [32]
    uint32_t * rows = (uint32_t *) (cb + 32);                 // [64][row_words]
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int j0 = blockIdx.x * 64, nc = min(64, n_kv - j0);
    for (int i = tid; i < rd * rd; i += blockDim.x) {
        const int m = i / rd, k = i % rd;
        Us[k * rd + m] = U_f16 ? __half2float(((const half *) U)[i]) : ((const float *) U)[i];
    }
    if (tid < 16) { cb[tid] = cbl_g[tid]; cb[16 + tid] = cbr_g[tid]; }
    for (int i = tid; i < 64 * row_words; i += blockDim.x) {
        const int jl = i / row_words, w = i % row_words;
        rows[i] = jl < nc ? ((const uint32_t *) (pk + (int64_t) (j0 + jl) * nb_cell))[w] : 0u;
    }
    __syncthreads();
    for (int i = tid; i < nc * (Ep - rd); i += blockDim.x) {  // latent + padding
        const int jl = i / (Ep - rd), e = i % (Ep - rd);
        float v = 0.0f;
        if (e < r) {
            const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
            half nlh; memcpy(&nlh, row + 4 + 4 * G, 2);
            float nl = __half2float(nlh); nl = nl > 1e-12f ? nl : 1e-12f;
            const uint8_t b = row[4 + 4 * G + 4 + rd / 2 + e / 2];
            v = cb[(e & 1) ? (b >> 4) : (b & 15)] * nl;
        }
        Kd[(int64_t) (j0 + jl) * Ep + rd + e] = __float2half(v);
    }
    if (rmsd) {                                                 // per-group key rms, 16-byte rows [n_kv][4]
        for (int i = tid; i < nc * 4; i += blockDim.x) {
            const int jl = i / 4, g = i % 4;
            float v = 1.0f;
            if (g < G) { memcpy(&v, (const uint8_t *) (rows + jl * row_words) + 4 + 4 * g, 4); v = v > 1e-6f ? v : 1e-6f; }
            rmsd[(int64_t) (j0 + jl) * 4 + g] = v;
        }
    }
    for (int jl = warp; jl < nc; jl += 8) {                   // RoPE key, warp per cell
        const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
        int32_t pos; memcpy(&pos, row, 4);
        half nrh; memcpy(&nrh, row + 4 + 4 * G + 2, 2);
        float nr = __half2float(nrh); nr = nr > 1e-12f ? nr : 1e-12f;
        float * rw = raw + warp * rd;
        for (int k = lane; k < rd; k += WARP_SIZE) {
            const uint8_t b = row[4 + 4 * G + 4 + k / 2];
            rw[k] = cb[16 + ((k & 1) ? (b >> 4) : (b & 15))] * nr;
        }
        __syncwarp();
        for (int i = lane; i < rd / 2; i += WARP_SIZE) {
            float x0 = 0.0f, x1 = 0.0f;
            for (int k = 0; k < rd; ++k) {
                x0 += Us[k * rd + i] * rw[k];
                x1 += Us[k * rd + i + rd / 2] * rw[k];
            }
            const float th = (float) pos * powf(theta_scale, (float) i);
            const float cs = cosf(th), sn = sinf(th);
            Kd[(int64_t) (j0 + jl) * Ep + i]          = __float2half(x0 * cs - x1 * sn);
            Kd[(int64_t) (j0 + jl) * Ep + i + rd / 2] = __float2half(x0 * sn + x1 * cs);
        }
        __syncwarp();
    }
}

// One block per query token: rows = its 16 heads. Online softmax over 64-cell key tiles read from Kd (L2-resident),
// S = Q K^T and O += P C on tensor cores; tiles the mask hides entirely are skipped.
template <typename mask_t>
static __global__ void __launch_bounds__(256) merc_tq_attn_prefill(
        const float * q, int64_t q_nb1, int64_t q_nb2, const half * Kd, const char * pk, int64_t nb_cell,
        const mask_t * mask, int64_t mask_nb1, float * dst, int64_t d_nb1, int64_t d_nb2,
        int n_kv, int G, int r, int rd, float scale) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    const int E = rd + r, Ep = merc_pad16(E), rp = merc_pad16(r);
    half  * Qs = (half *) smem_raw;                          // [16][Ep]
    float * Os = (float *) (Qs + MERC_ATTN_H * Ep);          // [16][rp] running output
    float * Ss = Os + MERC_ATTN_H * rp;                      // [16][64]
    half  * Ps = (half *) (Ss + MERC_ATTN_H * 64);           // [16][64]
    float * st = (float *) (Ps + MERC_ATTN_H * 64);          // [16] m, [16] l, [16] alpha, [1] tile live
    const int t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int rep = MERC_ATTN_H / G;
    for (int i = tid; i < MERC_ATTN_H * Ep; i += blockDim.x) {
        const int h = i / Ep, e = i % Ep;
        Qs[i] = __float2half(e < E ? *(const float *) ((const char *) q + t * q_nb1 + h * q_nb2 + e * sizeof(float)) : 0.0f);
    }
    for (int i = tid; i < MERC_ATTN_H * rp; i += blockDim.x) Os[i] = 0.0f;
    if (tid < MERC_ATTN_H) { st[tid] = -INFINITY; st[16 + tid] = 0.0f; }
    __syncthreads();
    for (int j0 = 0; j0 < n_kv; j0 += 64) {
        const int nc = min(64, n_kv - j0);
        if (tid == 0) st[48] = 0.0f;
        __syncthreads();
        if (tid < nc) {                                     // any visible cell in this tile?
            const float m = mask ? (float) mask[t * mask_nb1 + j0 + tid] : 0.0f;
            if (m != -INFINITY) st[48] = 1.0f;
        }
        __syncthreads();
        if (st[48] == 0.0f) continue;
        if (warp < 4) {                                     // S: 4 column tiles of 16 cells
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            if (j0 + warp * 16 < n_kv) {
                for (int e0 = 0; e0 < Ep; e0 += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fa, Qs + e0, Ep);
                    wmma::load_matrix_sync(fb, Kd + (int64_t) (j0 + warp * 16) * Ep + e0, Ep);
                    wmma::mma_sync(acc, fa, fb, acc);
                }
            }
            wmma::store_matrix_sync(Ss + warp * 16, acc, 64, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < MERC_ATTN_H * 64; i += blockDim.x) {
            const int h = i / 64, jl = i % 64;
            float s = -INFINITY;
            if (jl < nc) {
                const uint8_t * row = (const uint8_t *) (pk + (int64_t) (j0 + jl) * nb_cell);
                float rms; memcpy(&rms, row + 4 + 4 * (h / rep), 4);
                const float m = mask ? (float) mask[t * mask_nb1 + j0 + jl] : 0.0f;
                s = Ss[i] / (rms > 1e-6f ? rms : 1e-6f) * scale + m;
            }
            Ss[i] = s;
        }
        __syncthreads();
        for (int h = warp; h < MERC_ATTN_H; h += 8) {        // online softmax, warp per head
            float mx = -INFINITY;
            for (int jl = lane; jl < 64; jl += WARP_SIZE) mx = fmaxf(mx, Ss[h * 64 + jl]);
            mx = warp_reduce_max(mx);
            const float m_old = st[h], m_new = fmaxf(m_old, mx);
            float l = 0.0f;
            for (int jl = lane; jl < 64; jl += WARP_SIZE) {
                const float p = m_new == -INFINITY ? 0.0f : expf(Ss[h * 64 + jl] - m_new);
                Ps[h * 64 + jl] = __float2half(p);
                l += p;
            }
            l = warp_reduce_sum(l);
            if (lane == 0) {
                const float a = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                st[32 + h] = a; st[16 + h] = st[16 + h] * a + l; st[h] = m_new;
            }
        }
        __syncthreads();
        for (int i = tid; i < MERC_ATTN_H * rp; i += blockDim.x) Os[i] *= st[32 + i / rp];
        __syncthreads();
        for (int n0 = warp * 16; n0 < rp; n0 += 8 * 16) {     // O += P C
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, Os + n0, rp, wmma::mem_row_major);
            for (int k0 = 0; k0 < 64; k0 += 16) {
                if (j0 + k0 >= n_kv) break;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, Ps + k0, 64);
                wmma::load_matrix_sync(fb, Kd + (int64_t) (j0 + k0) * Ep + rd + n0, Ep);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(Os + n0, acc, rp, wmma::mem_row_major);
        }
        __syncthreads();
    }
    for (int i = tid; i < MERC_ATTN_H * r; i += blockDim.x) {
        const int h = i / r, e = i % r;
        const float l = st[16 + h];
        *(float *) ((char *) dst + t * d_nb1 + h * d_nb2 + e * sizeof(float)) = l > 0.0f ? Os[h * rp + e] / l : 0.0f;
    }
}

// Shared-memory prefill: 4 query tokens per block (64 rows = token x head), 16 warps. Q stays on chip; each 32-cell
// key tile is staged with coalesced cp.async loads, so every tensor-core operand comes from shared memory. S = Q K^T
// is split over the key width between two warp groups (8 tiles x 2 halves = 16 warps) and summed in smem; the
// softmax maps one lane to one cell; O = diag(alpha) O + P C stays in registers (C = latent part of the K tile).
#include "cp-async.cuh"
#define MSP_TQ   4
#define MSP_ROWS (MSP_TQ * MERC_ATTN_H)
#define MSP_KT   32
#define MSP_THR  512
#define MSP_MAXF 11
#define MSP_PS   40                              // P row stride (halves): 80 bytes, an odd number of 16-byte units

static __host__ __device__ inline int msp_stride(int Ep) { return (Ep / 8) % 2 == 0 ? Ep + 8 : Ep; }

// Double-buffered variant: 16-cell key tiles, the next tile's cp.async issued before the current one is computed.
// S [64 x 16] = 4 tiles, the key width split in 4 quarters -> 16 warps; partials summed in the softmax pass.
#define MDB_KT 16
#define MDB_PS 24                                // P row stride (halves): 48 bytes = 3 x 16

template <typename mask_t>
static __global__ void __launch_bounds__(MSP_THR) merc_tq_attn_prefill_smem(
        const float * q, int64_t q_nb1, int64_t q_nb2, const half * Kd, const float * rmsd,
        const mask_t * mask, int64_t mask_nb1, int mask_ne0, bool mask_al16, float * dst, int64_t d_nb1, int64_t d_nb2,
        int T, int n_kv, int G, int r, int rd, float scale) {
    using namespace nvcuda;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    const int E = rd + r, Ep = merc_pad16(E), rp = merc_pad16(r);
    const int Es = msp_stride(Ep);
    half  * Qs = (half *) smem_raw;                               // [ROWS][Es]
    half  * Kb = Qs + MSP_ROWS * Es;                              // [2][KT][Es]
    float * Sp = (float *) (Kb + 2 * MDB_KT * Es);                // [4][ROWS][KT] partial scores
    half  * Ps = (half *) (Sp + 4 * MSP_ROWS * MDB_KT);           // [ROWS][MDB_PS]
    float * mrow = (float *) (Ps + MSP_ROWS * MDB_PS);
    float * lrow = mrow + MSP_ROWS, * arow = lrow + MSP_ROWS, * map = arow + MSP_ROWS;
    float * rmst = map + 512;                                     // [2][KT][4]
    mask_t * mskt = (mask_t *) (rmst + 2 * MDB_KT * 4);           // [2][TQ][KT] (raw mask type)
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int nwarps = MSP_THR / WARP_SIZE;
    const int rep = MERC_ATTN_H / G;
    const int t0 = blockIdx.x * MSP_TQ;
    const int nct = rp / 16, nfrag = MSP_TQ * nct;
    const int nks = Ep / 16;
    const int segs = Ep / 8;
    // causal: the last key any of this block's tokens can see (mask decides exactly; this only bounds the loop)
    int jend = n_kv;
    __shared__ int jmax, any_rescale;
    if (tid == 0) jmax = -1;
    __syncthreads();
    if (mask) {
        const int tl = min(T - 1, t0 + MSP_TQ - 1);
        for (int j = tid; j < n_kv; j += blockDim.x) {
            if ((float) mask[tl * mask_nb1 + j] != -INFINITY) atomicMax(&jmax, j);
        }
        __syncthreads();
        jend = jmax + 1;
    }

    for (int i = tid; i < MSP_ROWS * Ep; i += blockDim.x) {
        const int row = i / Ep, e = i % Ep, t = t0 + row / MERC_ATTN_H, h = row % MERC_ATTN_H;
        Qs[row * Es + e] = __float2half(e < E && t < T ? *(const float *) ((const char *) q + t * q_nb1 + h * q_nb2 + e * sizeof(float)) : 0.0f);
    }
    for (int i = tid; i < 256; i += blockDim.x) { map[i] = (float) (i / 16); map[256 + i] = (float) (i % 16); }
    if (tid < MSP_ROWS) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }

    auto stage = [&](int j0, int b) {                             // key tile + its mask and rms
        half * Ks = Kb + b * MDB_KT * Es;
        for (int i = tid; i < MDB_KT * segs; i += MSP_THR) {
            const int cell = i / segs, sg = i % segs;
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(Ks + cell * Es + sg * 8), Kd + (int64_t) (j0 + cell) * Ep + sg * 8);
        }
        constexpr int mseg = MDB_KT * (int) sizeof(mask_t) / 16;    // 16-byte segments per token's mask row
        if (tid < MDB_KT) {                                        // rms rows (16 bytes per cell)
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(rmst + b * MDB_KT * 4 + tid * 4), rmsd + (int64_t) (j0 + tid) * 4);
        } else if (mask && tid < MDB_KT + MSP_TQ * mseg) {          // mask rows of the block's tokens
            const int i = tid - MDB_KT, tq = i / mseg, sg = i % mseg, t = min(t0 + tq, T - 1);
            constexpr int cps = 16 / (int) sizeof(mask_t);          // cells per 16-byte segment
            mask_t * dstm = mskt + b * MSP_TQ * MDB_KT + tq * MDB_KT + sg * cps;
            const mask_t * srcm = mask + t * mask_nb1 + j0 + sg * cps;
            if (mask_al16 && j0 + (sg + 1) * cps <= mask_ne0) {
                cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dstm), srcm);
            } else {                                                // row tail / unaligned rows: element by element
                for (int c = 0; c < cps; ++c) dstm[c] = j0 + sg * cps + c < mask_ne0 ? srcm[c] : (mask_t) -INFINITY;
            }
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf, O[MSP_MAXF];
    __syncthreads();
    wmma::load_matrix_sync(rowf, map, 16, wmma::mem_row_major);
    const int ra = (int) rowf.x[0];
    int rb = ra;                                                  // every fragment element lies in one of two rows
#pragma unroll
    for (int i = 0; i < rowf.num_elements; ++i) if ((int) rowf.x[i] != ra) rb = (int) rowf.x[i];
#pragma unroll
    for (int k = 0; k < MSP_MAXF; ++k) wmma::fill_fragment(O[k], 0.0f);

    if (jend > 0) stage(0, 0);
    for (int j0 = 0, it = 0; j0 < jend; j0 += MDB_KT, ++it) {
        const int b = it & 1;
        cp_async_wait_all();
        __syncthreads();                                          // tile b (and its mask/rms) visible to all
        if (j0 + MDB_KT < jend) stage(j0 + MDB_KT, b ^ 1);        // prefetch the next tile
        if (tid == 0) any_rescale = 0;
        const half * Ks = Kb + b * MDB_KT * Es;
        {                                                         // S partial: row tile = warp % 4, key-width quarter = warp / 4
            const int rt = warp % 4, qq = warp / 4;
            const int k_lo = nks * qq / 4, k_hi = nks * (qq + 1) / 4;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int ks = k_lo; ks < k_hi; ++ks) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, Qs + rt * 16 * Es + ks * 16, Es);
                wmma::load_matrix_sync(fb, Ks + ks * 16, Es);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(Sp + qq * MSP_ROWS * MDB_KT + rt * 16 * MDB_KT, acc, MDB_KT, wmma::mem_row_major);
        }
        __syncthreads();
        for (int rr = warp * 2; rr < MSP_ROWS; rr += nwarps * 2) {   // two rows per warp: lanes 0-15 / 16-31
            const int row = rr + lane / 16, jl = lane % 16;
            const int h = row % MERC_ATTN_H, tq = row / MERC_ATTN_H;
            const int t = t0 + tq;
            const float m = (t >= T || j0 + jl >= n_kv) ? -INFINITY : (mask ? (float) mskt[b * MSP_TQ * MDB_KT + tq * MDB_KT + jl] : 0.0f);
            float sv = -INFINITY;
            if (m != -INFINITY) {
                const int o = row * MDB_KT + jl, st = MSP_ROWS * MDB_KT;
                sv = (Sp[o] + Sp[st + o] + Sp[2 * st + o] + Sp[3 * st + o]) / rmst[b * MDB_KT * 4 + jl * 4 + h / rep] * scale + m;
            }
            float mx = sv;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 16));
            // lazy rescale: keep the reference max m_ref unless the row max exceeds it by more than 8 (p <= e^8 fits f16)
            const float m_ref = mrow[row];
            const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.0f);
            const float m_use = bump ? mx : m_ref;
            const float pv = m_use == -INFINITY ? 0.0f : expf(sv - m_use);
            Ps[row * MDB_PS + jl] = __float2half(pv);
            float l = pv;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffff, l, off, 16);
            __syncwarp();
            if (jl == 0) {
                const float a = bump ? (m_ref == -INFINITY ? 0.0f : expf(m_ref - m_use)) : 1.0f;
                arow[row] = a; lrow[row] = lrow[row] * a + l; mrow[row] = m_use;
                if (bump && m_ref != -INFINITY) any_rescale = 1;
            }
        }
        __syncthreads();
        const bool resc = any_rescale != 0;
        float al[MSP_TQ][2];                                      // this thread's two accumulator rows per row tile
        if (resc) {
#pragma unroll
            for (int rt = 0; rt < MSP_TQ; ++rt) { al[rt][0] = arow[rt * 16 + ra]; al[rt][1] = arow[rt * 16 + rb]; }
        }
#pragma unroll
        for (int k = 0; k < MSP_MAXF; ++k) {                      // O = diag(alpha) O + P C
            const int f = warp + nwarps * k;
            if (f < nfrag) {
                const int rt = f / nct, ct = f % nct;
                if (resc) {
#pragma unroll
                    for (int i = 0; i < O[k].num_elements; ++i) O[k].x[i] *= (rowf.x[i] == (float) ra) ? al[rt][0] : al[rt][1];
                }
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, Ps + rt * 16 * MDB_PS, MDB_PS);
                wmma::load_matrix_sync(fb, Ks + rd + ct * 16, Es);
                wmma::mma_sync(O[k], fa, fb, O[k]);
            }
        }
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> colf;
    wmma::load_matrix_sync(colf, map + 256, 16, wmma::mem_row_major);
#pragma unroll
    for (int k = 0; k < MSP_MAXF; ++k) {
        const int f = warp + nwarps * k;
        if (f < nfrag) {
            const int rt = f / nct, ct = f % nct;
#pragma unroll
            for (int i = 0; i < O[k].num_elements; ++i) {
                const int row = rt * 16 + (int) rowf.x[i], col = ct * 16 + (int) colf.x[i];
                const int t = t0 + row / MERC_ATTN_H, h = row % MERC_ATTN_H;
                if (t < T && col < r) {
                    const float l = lrow[row];
                    *(float *) ((char *) dst + t * d_nb1 + h * d_nb2 + col * sizeof(float)) = l > 0.0f ? O[k].x[i] / l : 0.0f;
                }
            }
        }
    }
}

void ggml_cuda_op_merc_tq_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * pk = dst->src[1], * un = dst->src[4], * mask = dst->src[5];
    const int r = ggml_get_op_params_i32(dst, 0), rd = ggml_get_op_params_i32(dst, 1), G = ggml_get_op_params_i32(dst, 2);
    const float scale = ggml_get_op_params_f32(dst, 3), freq_base = ggml_get_op_params_f32(dst, 4);
    const int T = (int) q->ne[1], H = (int) q->ne[2];
    const int n_kv = (int) (pk->ne[1] * pk->ne[2]);
    GGML_ASSERT(H == MERC_ATTN_H && rd % 16 == 0);
    GGML_ASSERT(un->type == GGML_TYPE_F32 || un->type == GGML_TYPE_F16);
    if (T == 0 || n_kv == 0) {
        return;
    }
    const int64_t nb_cell = pk->ne[1] == 1 ? pk->nb[2] : pk->nb[1];
    if (T > 8) {                                            // prefill: decode once to f16, query-tiled flash attention
        const int Ep_ = merc_pad16(rd + r), rp_ = merc_pad16(r);
        const int n_kv64 = (n_kv + 63) / 64 * 64;
        ggml_cuda_pool_alloc<half> kd(ctx.pool(), (size_t) n_kv64 * Ep_);
        ggml_cuda_pool_alloc<float> rmsd(ctx.pool(), (size_t) n_kv64 * 4);
        CUDA_CHECK(cudaMemsetAsync(rmsd.get(), 0, sizeof(float) * (size_t) n_kv64 * 4, ctx.stream()));
        CUDA_CHECK(cudaMemsetAsync(kd.get(), 0, sizeof(half) * (size_t) n_kv64 * Ep_, ctx.stream()));
        const size_t smem_d = sizeof(float) * (rd * rd + 8 * rd + 32) + sizeof(uint32_t) * 64 * pk->ne[0];
        CUDA_CHECK(cudaFuncSetAttribute(merc_tq_decode_f16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_d));
        merc_tq_decode_f16<<<(n_kv + 63) / 64, 256, smem_d, ctx.stream()>>>((const char *) pk->data, nb_cell, (int) pk->ne[0],
            (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, un->data, un->type == GGML_TYPE_F16,
            kd.get(), Ep_, n_kv, G, r, rd, powf(freq_base, -2.0f / rd), rmsd.get());
        {
            GGML_ASSERT(MSP_TQ * (rp_ / 16) <= MSP_MAXF * (MSP_THR / WARP_SIZE));
            const size_t smem_s = sizeof(half) * (MSP_ROWS + 2 * MDB_KT) * msp_stride(Ep_) + sizeof(float) * 4 * MSP_ROWS * MDB_KT +
                                  sizeof(half) * MSP_ROWS * MDB_PS + sizeof(float) * (3 * MSP_ROWS + 512 + 2 * MDB_KT * 4 + 2 * MSP_TQ * MDB_KT);
            GGML_ASSERT(G <= 4);
            const bool mal16 = mask && mask->nb[1] % 16 == 0 && (uintptr_t) mask->data % 16 == 0;
            const int nb = (T + MSP_TQ - 1) / MSP_TQ;
            if (mask == nullptr || mask->type == GGML_TYPE_F16) {
                CUDA_CHECK(cudaFuncSetAttribute(merc_tq_attn_prefill_smem<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s));
                merc_tq_attn_prefill_smem<half><<<nb, MSP_THR, smem_s, ctx.stream()>>>((const float *) q->data, q->nb[1], q->nb[2],
                    kd.get(), rmsd.get(), mask ? (const half *) mask->data : nullptr, mask ? mask->nb[1] / sizeof(half) : 0, mask ? (int) mask->ne[0] : 0, mal16,
                    (float *) dst->data, dst->nb[1], dst->nb[2], T, n_kv, G, r, rd, scale);
            } else {
                CUDA_CHECK(cudaFuncSetAttribute(merc_tq_attn_prefill_smem<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s));
                merc_tq_attn_prefill_smem<float><<<nb, MSP_THR, smem_s, ctx.stream()>>>((const float *) q->data, q->nb[1], q->nb[2],
                    kd.get(), rmsd.get(), (const float *) mask->data, mask->nb[1] / sizeof(float), (int) mask->ne[0], mal16,
                    (float *) dst->data, dst->nb[1], dst->nb[2], T, n_kv, G, r, rd, scale);
            }
            return;
        }
        return;
    }
    const int n_chunks = (n_kv + MERC_ATTN_CHUNK - 1) / MERC_ATTN_CHUNK;
    const int Ep = merc_pad16(rd + r), rp = merc_pad16(r);
    ggml_cuda_pool_alloc<float> part_o(ctx.pool(), (size_t) T * n_chunks * MERC_ATTN_H * rp);
    ggml_cuda_pool_alloc<float> part_ml(ctx.pool(), (size_t) T * n_chunks * 2 * MERC_ATTN_H);
    const size_t smem = sizeof(half) * (size_t) (MERC_ATTN_CHUNK + MERC_ATTN_H) * Ep + sizeof(float) * rd * rd +
                        sizeof(float) * MERC_ATTN_H * MERC_ATTN_CHUNK + sizeof(half) * MERC_ATTN_H * MERC_ATTN_CHUNK +
                        sizeof(float) * ((MERC_ATTN_THREADS / WARP_SIZE) * rd + 32) +
                        sizeof(uint32_t) * (size_t) MERC_ATTN_CHUNK * pk->ne[0];
    const float theta_scale = powf(freq_base, -2.0f / rd);
    const int n_sm = ggml_cuda_info().devices[ctx.device].nsm;
    const dim3 grid(std::min(n_chunks, std::max(1, 2 * n_sm / std::max(T, 1))), T);
    cudaStream_t st = ctx.stream();
    const char * pkd = (const char *) pk->data;
    const float * cbl = (const float *) dst->src[2]->data, * cbr = (const float *) dst->src[3]->data;
    if (mask == nullptr || mask->type == GGML_TYPE_F16) {
        CUDA_CHECK(cudaFuncSetAttribute(merc_tq_attn_chunk<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        merc_tq_attn_chunk<half><<<grid, MERC_ATTN_THREADS, smem, st>>>((const float *) q->data, q->nb[1], q->nb[2], pkd,
            nb_cell, (int) pk->ne[0], cbl, cbr, un->data, un->type == GGML_TYPE_F16, mask ? (const half *) mask->data : nullptr,
            mask ? mask->nb[1] / sizeof(half) : 0, part_o.get(), part_ml.get(), n_kv, n_chunks, G, r, rd, scale, theta_scale);
    } else {
        CUDA_CHECK(cudaFuncSetAttribute(merc_tq_attn_chunk<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        merc_tq_attn_chunk<float><<<grid, MERC_ATTN_THREADS, smem, st>>>((const float *) q->data, q->nb[1], q->nb[2], pkd,
            nb_cell, (int) pk->ne[0], cbl, cbr, un->data, un->type == GGML_TYPE_F16, (const float *) mask->data,
            mask->nb[1] / sizeof(float), part_o.get(), part_ml.get(), n_kv, n_chunks, G, r, rd, scale, theta_scale);
    }
    merc_tq_attn_reduce<<<dim3(H, T), 256, 0, st>>>(part_o.get(), part_ml.get(), (float *) dst->data, dst->nb[1],
                                                     dst->nb[2], n_chunks, r, rp);
}

// ---------------------------------------------------------------------------------------------------------------
// GGML_OP_MERC_TQ_EXPAND: expanded per-group K / V (f16) for prefill flash attention. Decode the packed cache once
// (merc_tq_decode_f16: [rope key | latent] f16 + rms), then per chunk of cells two GEMMs on the latent (key features
// [nres shared | G x nope], values [G x D]) and an assembly pass that writes K_g / rms_g and V_g.

static __global__ void merc_tq_expand_assemble(
        const half * Kd, int Ep, const float * rmsd, const half * KN, const half * VN, half * dst, int64_t d_nb1,
        int j0, int nc, int G, int rd, int nres, int nope, int D) {
    const int jl = blockIdx.x;
    if (jl >= nc) return;
    const int j = j0 + jl, Ek = rd + nres + nope, nkn = nres + G * nope;
    half * y = (half *) ((char *) dst + (int64_t) j * d_nb1);
    const half * kd = Kd + (int64_t) j * Ep;
    const half * kn = KN + (int64_t) jl * nkn;
    const half * vn = VN + (int64_t) jl * G * D;
    for (int i = threadIdx.x; i < G * Ek; i += blockDim.x) {
        const int g = i / Ek, e = i % Ek;
        const float inv = 1.0f / rmsd[(int64_t) j * 4 + g];
        float v;
        if (e < rd)             v = __half2float(kd[e]);
        else if (e < rd + nres) v = __half2float(kn[e - rd]);
        else                    v = __half2float(kn[nres + g * nope + (e - rd - nres)]);
        y[i] = __float2half(v * inv);
    }
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        y[G * Ek + i] = vn[i];
    }
}

static __global__ void merc_f32_to_f16(const float * x, half * y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

void ggml_cuda_op_merc_tq_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * pk = dst->src[0], * un = dst->src[3], * ku = dst->src[4], * vu = dst->src[5];
    const int r = ggml_get_op_params_i32(dst, 0), rd = ggml_get_op_params_i32(dst, 1), G = ggml_get_op_params_i32(dst, 2);
    const float freq_base = ggml_get_op_params_f32(dst, 3);
    const int D = (int) vu->ne[1], nres = (G - 1) * rd, nope = D - rd, nkn = nres + G * nope;
    const int n_kv = (int) (pk->ne[1] * pk->ne[2]);
    if (n_kv == 0) return;
    GGML_ASSERT(G <= 4);
    cudaStream_t st = ctx.stream();
    const int64_t nb_cell = pk->ne[1] == 1 ? pk->nb[2] : pk->nb[1];
    const int Ep = merc_pad16(rd + r);
    const int n_kv64 = (n_kv + 63) / 64 * 64;
    ggml_cuda_pool_alloc<half>  kd(ctx.pool(), (size_t) n_kv64 * Ep);
    ggml_cuda_pool_alloc<float> rmsd(ctx.pool(), (size_t) n_kv64 * 4);
    const size_t smem_d = sizeof(float) * (rd * rd + 8 * rd + 32) + sizeof(uint32_t) * 64 * pk->ne[0];
    CUDA_CHECK(cudaFuncSetAttribute(merc_tq_decode_f16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_d));
    merc_tq_decode_f16<<<(n_kv + 63) / 64, 256, smem_d, st>>>((const char *) pk->data, nb_cell, (int) pk->ne[0],
        (const float *) dst->src[1]->data, (const float *) dst->src[2]->data, un->data, un->type == GGML_TYPE_F16,
        kd.get(), Ep, n_kv, G, r, rd, powf(freq_base, -2.0f / rd), rmsd.get());
    // weights as f16 (converted once per call when stored as f32)
    ggml_cuda_pool_alloc<half> ku16(ctx.pool()), vu16(ctx.pool());
    const half * KU = (const half *) ku->data, * VU = (const half *) vu->data;
    if (ku->type == GGML_TYPE_F32) {
        ku16.alloc((size_t) r * nkn);
        merc_f32_to_f16<<<((int64_t) r * nkn + 255) / 256, 256, 0, st>>>((const float *) ku->data, ku16.get(), (int64_t) r * nkn);
        KU = ku16.get();
    }
    if (vu->type == GGML_TYPE_F32) {
        vu16.alloc((size_t) r * D * G);
        merc_f32_to_f16<<<((int64_t) r * D * G + 255) / 256, 256, 0, st>>>((const float *) vu->data, vu16.get(), (int64_t) r * D * G);
        VU = vu16.get();
    }
    const int CH = 8192;
    ggml_cuda_pool_alloc<half> kn(ctx.pool(), (size_t) CH * nkn), vn(ctx.pool(), (size_t) CH * G * D);
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), st));
    const float alpha = 1.0f, beta = 0.0f;
    for (int j0 = 0; j0 < n_kv; j0 += CH) {
        const int nc = std::min(CH, n_kv - j0);
        const half * C = kd.get() + (int64_t) j0 * Ep + rd;           // latent of cell j0, column-major r x nc, ld Ep
        // KN (nkn x nc) = KU^T (nkn x r) * C (r x nc)
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, nkn, nc, r,
            &alpha, KU, CUDA_R_16F, r, C, CUDA_R_16F, Ep, &beta, kn.get(), CUDA_R_16F, nkn,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, G * D, nc, r,
            &alpha, VU, CUDA_R_16F, r, C, CUDA_R_16F, Ep, &beta, vn.get(), CUDA_R_16F, G * D,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        merc_tq_expand_assemble<<<nc, 256, 0, st>>>(kd.get(), Ep, rmsd.get(), kn.get(), vn.get(), (half *) dst->data,
            dst->nb[1], j0, nc, G, rd, nres, nope, D);
    }
}
