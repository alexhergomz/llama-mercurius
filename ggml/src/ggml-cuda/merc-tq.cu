#include "merc-tq.cuh"
#include "cp-async.cuh"

// 16-byte global -> shared copy: cp.async on Ampere and newer, a plain vector copy before (Volta / Turing)
static __device__ __forceinline__ void merc_copy16(void * dst_smem, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(dst_smem), src);
#else
    *(int4 *) dst_smem = *(const int4 *) src;
#endif // CP_ASYNC_AVAILABLE
}
static __device__ __forceinline__ void merc_copy_wait() {
#ifdef CP_ASYNC_AVAILABLE
    cp_async_wait_all();
#endif // CP_ASYNC_AVAILABLE
}

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

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)   // nvcuda::wmma (tensor cores): NVIDIA only
#include <mma.h>
#endif

#define MERC_ATTN_CHUNK   64
#define MERC_ATTN_THREADS 512
#define MERC_ATTN_H       16                     // query heads: one 16-row MMA tile

static __host__ __device__ inline int merc_pad16(int x) { return (x + 15) / 16 * 16; }
static __host__ __device__ inline int msp_stride(int Ep) { return (Ep / 8) % 2 == 0 ? Ep + 8 : Ep; }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)   // nvcuda::wmma (tensor cores): NVIDIA only
#define MERC_QF_MAX 12                           // Q fragments per warp: a quarter of Ep / 16 (Ep <= 768)

// Split-KV (flash-decoding) attention over the packed cache for a few query tokens. Block (b, t) walks a contiguous
// range of 64-cell chunks with a running softmax per head (max, sum, fp32 accumulators in MMA fragments; lazy rescale
// as in the prefill kernel) and writes ONE unnormalized partial per block; merc_tq_attn_reduce merges the few partials.
// The next chunk's packed rows are loaded into registers while the current chunk is decoded and attended.
template <typename mask_t>
static __global__ void __launch_bounds__(MERC_ATTN_THREADS) merc_tq_attn_split(
        const float * q, int64_t q_nb1, int64_t q_nb2, const char * pk, int64_t nb_cell, int row_words,
        const float * cbl_g, const float * cbr_g, const void * U, bool U_f16, const mask_t * mask, int64_t mask_nb1,
        float * part_o, float * part_ml, int n_kv, int n_chunks, int cpb, int G, int r, int rd, float scale,
        float theta_scale, bool rows_async) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_raw[];
    const int E = rd + r, Ep = merc_pad16(E), rp = merc_pad16(r);
    // shared-memory row strides padded to an odd number of 16-byte units (conflict-free MMA tile loads / stores)
    const int Es = msp_stride(Ep), Ul = rd + 8, Rl = rd + 4;
    constexpr int Sl = MERC_ATTN_CHUNK + 4, Pl = MERC_ATTN_CHUNK + 8;
    half  * Ks  = (half *) smem_raw;                          // [CHUNK][Es]   decoded keys (latent part = C)
    half  * Uh  = Ks + MERC_ATTN_CHUNK * Es;                  // [rd][Ul] R0^T as stored: Uh[m*Ul + k] = U[m][k]
    float * Ss  = (float *) (Uh + rd * Ul);                   // [4][16][Sl] partial scores; also [CHUNK][Rl] rope GEMM out
    half  * Ps  = (half *) (Ss + 4 * MERC_ATTN_H * Sl);       // [16][Pl] probabilities
    float * invf = (float *) (Ps + MERC_ATTN_H * Pl);         // [32] RoPE frequencies
    float * mrow = invf + 32, * lrow = mrow + MERC_ATTN_H, * arow = lrow + MERC_ATTN_H;
    int   * posj = (int *) (arow + MERC_ATTN_H);                       // [CHUNK] cell positions
    float * rmsj = (float *) (posj + MERC_ATTN_CHUNK);                 // [CHUNK][4] per-group key rms (clamped)
    uint32_t * rows = (uint32_t *) (rmsj + 4 * MERC_ATTN_CHUNK);       // [CHUNK][row_words] packed cells (16-aligned)
    half  * Qs  = Ks;                                         // [16][Es] queries, staged once before the loop
    float * map = Ss;                                         // [256] row index of each 16x16 element, before the loop
    __shared__ int any_rescale;
    const int t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int nwarps = MERC_ATTN_THREADS / WARP_SIZE;
    const int rep = MERC_ATTN_H / G;
    const int nct = rp / 16, nks = Ep / 16;
    const int c_begin = blockIdx.x * cpb, c_end = min(n_chunks, c_begin + cpb);
    const int ct_s = warp % 4, qq = warp / 4;                 // S: column tile of 16 cells, key-width quarter
    const int k_lo = nks * qq / 4, k_hi = nks * (qq + 1) / 4;

    for (int i = tid; i < MERC_ATTN_H * Ep; i += blockDim.x) {
        const int h = i / Ep, e = i % Ep;
        Qs[h * Es + e] = __float2half(e < E ? *(const float *) ((const char *) q + t * q_nb1 + h * q_nb2 + e * sizeof(float)) : 0.0f);
    }
    for (int i = tid; i < rd * rd; i += blockDim.x) {
        Uh[(i / rd) * Ul + i % rd] = U_f16 ? ((const half *) U)[i] : __float2half(((const float *) U)[i]);
    }
    if (tid < rd / 2) invf[tid] = powf(theta_scale, (float) tid);
    if (tid < MERC_ATTN_H) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }
    // codebooks in registers, read by warp shuffle: lanes 0-15 hold the latent centroids, lanes 16-31 the rope ones
    const float cb_r = lane < 16 ? cbl_g[lane] : cbr_g[lane - 16];
    const int code_off = 4 + 4 * G + 4;                       // byte offset of the rope codes in a row
    for (int i = tid; i < 256; i += blockDim.x) map[i] = (float) (i / 16);

    // packed rows of a chunk -> shared memory. With contiguous 16-byte-aligned rows: cp.async (16-byte segments, word
    // tail), issued once the previous chunk's rows are decoded so the copy overlaps the rest of that chunk's work
    auto stage = [&](int chunk) {
        const int j0 = chunk * MERC_ATTN_CHUNK, nc = min(MERC_ATTN_CHUNK, n_kv - j0);
        const char * src = pk + (int64_t) j0 * nb_cell;
        if (rows_async) {
            const int bytes = nc * row_words * 4, nseg = bytes / 16;
            for (int sg = tid; sg < nseg; sg += MERC_ATTN_THREADS) {
                merc_copy16(rows + 4 * sg, src + 16 * sg);
            }
            for (int w = 4 * nseg + tid; w < bytes / 4; w += MERC_ATTN_THREADS) rows[w] = ((const uint32_t *) src)[w];
        } else {
            for (int i = tid; i < nc * row_words; i += MERC_ATTN_THREADS) {
                const int jl = i / row_words, w = i % row_words;
                rows[i] = ((const uint32_t *) (src + (int64_t) jl * nb_cell))[w];
            }
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf, O[3];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fq[MERC_QF_MAX];   // this warp's Q quarter
    __syncthreads();
    wmma::load_matrix_sync(rowf, map, 16, wmma::mem_row_major);
#pragma unroll
    for (int u = 0; u < MERC_QF_MAX; ++u) if (k_lo + u < k_hi) wmma::load_matrix_sync(fq[u], Qs + (k_lo + u) * 16, Es);
    const int ra = (int) rowf.x[0];
    int rb = ra;                                                  // every fragment element lies in one of two rows
#pragma unroll
    for (int i = 0; i < rowf.num_elements; ++i) if ((int) rowf.x[i] != ra) rb = (int) rowf.x[i];
#pragma unroll
    for (int k = 0; k < 3; ++k) wmma::fill_fragment(O[k], 0.0f);

    if (c_begin < c_end) stage(c_begin);
    for (int chunk = c_begin; chunk < c_end; ++chunk) {
        const int j0 = chunk * MERC_ATTN_CHUNK;
        const int nc = min(MERC_ATTN_CHUNK, n_kv - j0);
        merc_copy_wait();
        __syncthreads();                                   // rows of this chunk in place; previous chunk consumed
        if (tid == 0) any_rescale = 0;

        // decode, one warp per cell: raw rope key (f16, into the rope columns) and latent (8 codes per 32-bit word,
        // one 16-byte store per 8 values); empty cells and padding are zero
        const int nlat8 = (Ep - rd) / 8;
        for (int jl = warp; jl < MERC_ATTN_CHUNK; jl += nwarps) {
            half * kr = Ks + jl * Es;
            if (jl >= nc) {
                for (int w = lane; w < Ep / 8; w += WARP_SIZE) ((uint4 *) kr)[w] = make_uint4(0, 0, 0, 0);
                continue;
            }
            const uint8_t * row = (const uint8_t *) (rows + jl * row_words);
            half nlh, nrh; memcpy(&nlh, row + 4 + 4 * G, 2); memcpy(&nrh, row + 4 + 4 * G + 2, 2);
            float nl = __half2float(nlh); nl = nl > 1e-12f ? nl : 1e-12f;
            float nr = __half2float(nrh); nr = nr > 1e-12f ? nr : 1e-12f;
            if (lane == 0) memcpy(posj + jl, row, 4);
            if (lane < 4) {
                float v = 1.0f;
                if (lane < G) { memcpy(&v, row + 4 + 4 * lane, 4); v = v > 1e-6f ? v : 1e-6f; }
                rmsj[jl * 4 + lane] = v;
            }
            for (int w0 = 0; w0 < rd / 8 + nlat8; w0 += WARP_SIZE) {       // warp-uniform trip count (shuffles)
                const int w = w0 + lane;
                const bool act = w < rd / 8 + nlat8;
                const bool lat = w >= rd / 8;
                const int e0 = lat ? 8 * (w - rd / 8) : 8 * w;      // first element of this group
                const int n  = lat ? r : rd;
                uint32_t codes = 0;
                if (act && e0 < n) codes = *(const uint32_t *) (row + code_off + (lat ? rd / 2 : 0) + e0 / 2);
                uint32_t o[4];                                  // 8 halves packed in registers
                const int cbo = lat ? 0 : 16;
                const float nn = lat ? nl : nr;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int c0 = (codes >> (8 * u)) & 15, c1 = (codes >> (8 * u + 4)) & 15;
                    const float v0 = __shfl_sync(0xffffffff, cb_r, c0 + cbo) * nn;
                    const float v1 = __shfl_sync(0xffffffff, cb_r, c1 + cbo) * nn;
                    o[u] = (uint32_t) __half_as_ushort(__float2half(e0 + 2 * u < n ? v0 : 0.0f)) |
                           (uint32_t) __half_as_ushort(__float2half(e0 + 2 * u + 1 < n ? v1 : 0.0f)) << 16;
                }
                if (act) *(uint4 *) (kr + (lat ? rd : 0) + e0) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();                                   // rows decoded: the buffer takes the next chunk
        if (chunk + 1 < c_end) stage(chunk + 1);
        // rope keys: [CHUNK x rd] raw x R0 (tensor cores, f32 out into Ss), then NeoX rotation by position
        {
            const int ntm = rd / 16;
            for (int tt = warp; tt < (MERC_ATTN_CHUNK / 16) * ntm; tt += nwarps) {
                const int ct = tt / ntm, mt = tt % ntm;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                wmma::fill_fragment(acc, 0.0f);
                for (int k0 = 0; k0 < rd; k0 += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fa, Ks + ct * 16 * Es + k0, Es);
                    wmma::load_matrix_sync(fb, Uh + mt * 16 * Ul + k0, Ul);        // B[k][m] = U[m][k]
                    wmma::mma_sync(acc, fa, fb, acc);
                }
                wmma::store_matrix_sync(Ss + ct * 16 * Rl + mt * 16, acc, Rl, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int i = tid; i < MERC_ATTN_CHUNK * (rd / 2); i += blockDim.x) {
            const int jl = i / (rd / 2), p = i % (rd / 2);
            if (jl >= nc) continue;
            const int pos = posj[jl];
            const float x0 = Ss[jl * Rl + p], x1 = Ss[jl * Rl + p + rd / 2];
            float sn, cs; sincosf((float) pos * invf[p], &sn, &cs);
            Ks[jl * Es + p]          = __float2half(x0 * cs - x1 * sn);
            Ks[jl * Es + p + rd / 2] = __float2half(x0 * sn + x1 * cs);
        }
        __syncthreads();

        // S = Q K^T: 4 column tiles of 16 cells x 4 key-width quarters, one warp each
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int u = 0; u < MERC_QF_MAX; ++u) {
                if (k_lo + u < k_hi) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fb, Ks + ct_s * 16 * Es + (k_lo + u) * 16, Es);
                    wmma::mma_sync(acc, fq[u], fb, acc);
                }
            }
            wmma::store_matrix_sync(Ss + qq * MERC_ATTN_H * Sl + ct_s * 16, acc, Sl, wmma::mem_row_major);
        }
        __syncthreads();

        // per head (one warp each): 1/rms, scale, mask, chunk max, lazy rescale against the running max, P, sum
        for (int h = warp; h < MERC_ATTN_H; h += nwarps) {
            float sv[MERC_ATTN_CHUNK / WARP_SIZE];
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < MERC_ATTN_CHUNK / WARP_SIZE; ++u) {
                const int jl = lane + u * WARP_SIZE;
                float s = -INFINITY;
                if (jl < nc) {
                    const float m = mask ? (float) mask[t * mask_nb1 + j0 + jl] : 0.0f;
                    if (m != -INFINITY) {
                        const int o = h * Sl + jl, st = MERC_ATTN_H * Sl;
                        s = (Ss[o] + Ss[st + o] + Ss[2 * st + o] + Ss[3 * st + o]) / rmsj[jl * 4 + h / rep] * scale + m;
                    }
                }
                sv[u] = s;
                mx = fmaxf(mx, s);
            }
            mx = warp_reduce_max(mx);
            const float m_ref = mrow[h];
            const bool bump = mx != -INFINITY && (m_ref == -INFINITY || mx > m_ref + 8.0f);
            const float m_use = bump ? mx : m_ref;
            float l = 0.0f;
#pragma unroll
            for (int u = 0; u < MERC_ATTN_CHUNK / WARP_SIZE; ++u) {
                const float p = m_use == -INFINITY || sv[u] == -INFINITY ? 0.0f : expf(sv[u] - m_use);
                Ps[h * Pl + lane + u * WARP_SIZE] = __float2half(p);
                l += p;
            }
            l = warp_reduce_sum(l);
            if (lane == 0) {
                const float a = bump ? (m_ref == -INFINITY ? 0.0f : expf(m_ref - m_use)) : 1.0f;
                arow[h] = a; lrow[h] = lrow[h] * a + l; mrow[h] = m_use;
                if (bump && m_ref != -INFINITY) any_rescale = 1;
            }
        }
        __syncthreads();

        // O = diag(alpha) O + P C: rp / 16 column tiles over the warps, K dim = 64 cells
        const bool resc = any_rescale != 0;
        const float a0 = resc ? arow[ra] : 1.0f, a1 = resc ? arow[rb] : 1.0f;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int ct = warp + nwarps * k;
            if (ct < nct) {
                if (resc) {
#pragma unroll
                    for (int i = 0; i < O[k].num_elements; ++i) O[k].x[i] *= (rowf.x[i] == (float) ra) ? a0 : a1;
                }
                for (int k0 = 0; k0 < MERC_ATTN_CHUNK; k0 += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fa, Ps + k0, Pl);
                    wmma::load_matrix_sync(fb, Ks + k0 * Es + rd + ct * 16, Es);
                    wmma::mma_sync(O[k], fa, fb, O[k]);
                }
            }
        }
    }
    // one partial per block: unnormalized O and (max, sum) per head
    const int64_t pidx = (int64_t) t * gridDim.x + blockIdx.x;
    float * o = part_o + pidx * MERC_ATTN_H * rp;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int ct = warp + nwarps * k;
        if (ct < nct) wmma::store_matrix_sync(o + ct * 16, O[k], rp, wmma::mem_row_major);
    }
    __syncthreads();
    if (tid < MERC_ATTN_H) {
        part_ml[pidx * 2 * MERC_ATTN_H + tid]               = mrow[tid];
        part_ml[pidx * 2 * MERC_ATTN_H + MERC_ATTN_H + tid] = lrow[tid];
    }
}

#endif

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

// Shared-memory prefill: 4 query tokens per block (64 rows = token x head), 16 warps. Q stays on chip; each 32-cell
// key tile is staged with coalesced cp.async loads, so every tensor-core operand comes from shared memory. S = Q K^T
// is split over the key width between two warp groups (8 tiles x 2 halves = 16 warps) and summed in smem; the
// softmax maps one lane to one cell; O = diag(alpha) O + P C stays in registers (C = latent part of the K tile).
#define MSP_TQ   4
#define MSP_ROWS (MSP_TQ * MERC_ATTN_H)
#define MSP_KT   32
#define MSP_THR  512
#define MSP_MAXF 11
#define MSP_PS   40                              // P row stride (halves): 80 bytes, an odd number of 16-byte units


// Double-buffered variant: 16-cell key tiles, the next tile's cp.async issued before the current one is computed.
// S [64 x 16] = 4 tiles, the key width split in 4 quarters -> 16 warps; partials summed in the softmax pass.
#define MDB_KT 16
#define MDB_PS 24                                // P row stride (halves): 48 bytes = 3 x 16

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)   // nvcuda::wmma (tensor cores): NVIDIA only
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
            merc_copy16(Ks + cell * Es + sg * 8, Kd + (int64_t) (j0 + cell) * Ep + sg * 8);
        }
        constexpr int mseg = MDB_KT * (int) sizeof(mask_t) / 16;    // 16-byte segments per token's mask row
        if (tid < MDB_KT) {                                        // rms rows (16 bytes per cell)
            merc_copy16(rmst + b * MDB_KT * 4 + tid * 4, rmsd + (int64_t) (j0 + tid) * 4);
        } else if (mask && tid < MDB_KT + MSP_TQ * mseg) {          // mask rows of the block's tokens
            const int i = tid - MDB_KT, tq = i / mseg, sg = i % mseg, t = min(t0 + tq, T - 1);
            constexpr int cps = 16 / (int) sizeof(mask_t);          // cells per 16-byte segment
            mask_t * dstm = mskt + b * MSP_TQ * MDB_KT + tq * MDB_KT + sg * cps;
            const mask_t * srcm = mask + t * mask_nb1 + j0 + sg * cps;
            if (mask_al16 && j0 + (sg + 1) * cps <= mask_ne0) {
                merc_copy16(dstm, srcm);
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
        merc_copy_wait();
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

#endif

void ggml_cuda_op_merc_tq_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("MERC_TQ_ATTN: tensor-core kernels are NVIDIA-only (supports_op reports false here)");
#else
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
    GGML_ASSERT(rd <= 64 && rd % 16 == 0 && G <= 4 && rp <= 3 * 16 * (MERC_ATTN_THREADS / WARP_SIZE) && Ep <= 4 * 16 * MERC_QF_MAX);
    const bool rows_async = nb_cell == 4 * pk->ne[0] && (uintptr_t) pk->data % 16 == 0;
    const size_t smem = sizeof(half) * (size_t) MERC_ATTN_CHUNK * msp_stride(Ep) + sizeof(half) * rd * (rd + 8) +
                        sizeof(float) * 4 * MERC_ATTN_H * (MERC_ATTN_CHUNK + 4) + sizeof(half) * MERC_ATTN_H * (MERC_ATTN_CHUNK + 8) +
                        sizeof(float) * (32 + 3 * MERC_ATTN_H) + sizeof(float) * 5 * MERC_ATTN_CHUNK +
                        sizeof(uint32_t) * (size_t) MERC_ATTN_CHUNK * pk->ne[0];
    GGML_ASSERT(MERC_ATTN_CHUNK * (rd + 4) <= 4 * MERC_ATTN_H * (MERC_ATTN_CHUNK + 4));   // rope GEMM output fits Ss
    const float theta_scale = powf(freq_base, -2.0f / rd);
    cudaStream_t st = ctx.stream();
    const char * pkd = (const char *) pk->data;
    const float * cbl = (const float *) dst->src[2]->data, * cbr = (const float *) dst->src[3]->data;
    const bool mf16 = mask == nullptr || mask->type == GGML_TYPE_F16;
    const void * kfn = mf16 ? (const void *) merc_tq_attn_split<half> : (const void *) merc_tq_attn_split<float>;
    CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    // split-KV grid: enough blocks to fill the GPU once (all tokens together), contiguous chunk ranges per block
    int occ = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kfn, MERC_ATTN_THREADS, smem));
    const int n_sm = ggml_cuda_info().devices[ctx.device].nsm;
    const int want = std::max(1, (std::max(occ, 1) * n_sm + T - 1) / T);
    const int cpb = (n_chunks + std::min(n_chunks, want) - 1) / std::min(n_chunks, want);
    const int nblk = (n_chunks + cpb - 1) / cpb;
    ggml_cuda_pool_alloc<float> part_o(ctx.pool(), (size_t) T * nblk * MERC_ATTN_H * rp);
    ggml_cuda_pool_alloc<float> part_ml(ctx.pool(), (size_t) T * nblk * 2 * MERC_ATTN_H);
    const dim3 grid(nblk, T);
    if (mf16) {
        merc_tq_attn_split<half><<<grid, MERC_ATTN_THREADS, smem, st>>>((const float *) q->data, q->nb[1], q->nb[2], pkd,
            nb_cell, (int) pk->ne[0], cbl, cbr, un->data, un->type == GGML_TYPE_F16, mask ? (const half *) mask->data : nullptr,
            mask ? mask->nb[1] / sizeof(half) : 0, part_o.get(), part_ml.get(), n_kv, n_chunks, cpb, G, r, rd, scale, theta_scale, rows_async);
    } else {
        merc_tq_attn_split<float><<<grid, MERC_ATTN_THREADS, smem, st>>>((const float *) q->data, q->nb[1], q->nb[2], pkd,
            nb_cell, (int) pk->ne[0], cbl, cbr, un->data, un->type == GGML_TYPE_F16, (const float *) mask->data,
            mask->nb[1] / sizeof(float), part_o.get(), part_ml.get(), n_kv, n_chunks, cpb, G, r, rd, scale, theta_scale, rows_async);
    }
    merc_tq_attn_reduce<<<dim3(H, T), 256, 0, st>>>(part_o.get(), part_ml.get(), (float *) dst->data, dst->nb[1],
                                                     dst->nb[2], nblk, r, rp);
#endif
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
    const half * vn = VN ? VN + (int64_t) jl * G * D : nullptr;
    for (int i = threadIdx.x; i < G * Ek; i += blockDim.x) {
        const int g = i / Ek, e = i % Ek;
        const float inv = 1.0f / rmsd[(int64_t) j * 4 + g];
        float v;
        if (e < rd)             v = __half2float(kd[e]);
        else if (e < rd + nres) v = __half2float(kn[e - rd]);
        else                    v = __half2float(kn[nres + g * nope + (e - rd - nres)]);
        y[i] = __float2half(v * inv);
    }
    if (vn) {
        for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
            y[G * Ek + i] = vn[i];
        }
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

// ---------------------------------------------------------------------------------------------------------------
// GGML_OP_MERC_TQ_PREFILL: expanded-form prefill attention over the packed cache in fixed-size slices. Per slice of
// MERC_PREFILL_SLICE cells: decode to f16 [rope key | latent] + rms, two GEMMs on the latent (keys written to a small
// chunk buffer and assembled per group with the rms; values written by the GEMM straight into the slice), then flash
// attention (MMA kernel, 448/256) on the slice. Slices are merged online by log-sum-exp (partial outputs: unnormalized
// numerator + (max, sum) per row, as in vLLM's chunked MLA prefill). Temporary memory is bounded by the slice size and
// the batch, not the context, and lives in this op's compute-buffer slot (see ggml_cuda_merc_tq_prefill_get_alloc_size).

#define MERC_PREFILL_SLICE 16384                   // cells per slice (multiple of FATTN_KQ_STRIDE = 256)
#define MERC_PREFILL_CH    8192                    // cells per key GEMM chunk

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_partial_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                                    float * partial_dst, float2 * partial_meta);
extern template void ggml_cuda_flash_attn_ext_mma_f16_case<448, 256, 16, 4>(ggml_backend_cuda_context &, ggml_tensor *);
extern template void ggml_cuda_flash_attn_ext_mma_f16_partial_case<448, 256, 16, 4>(
        ggml_backend_cuda_context &, ggml_tensor *, float *, float2 *);

// rows of r values (f16 or f32) -> f16 rows of kp >= r values, zero-filled
static __global__ void merc_pad_rows_f16(const void * x, bool x_f16, int r, int kp, int64_t nrows, half * y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * kp) return;
    const int64_t row = i / kp; const int e = (int) (i % kp);
    y[i] = e >= r ? __float2half(0.0f) : x_f16 ? ((const half *) x)[row * r + e] : __float2half(((const float *) x)[row * r + e]);
}

struct merc_prefill_layout {
    int64_t S, CH, Ep, kp, nkn, row_halves, nrows;
    size_t off_kd, off_rms, off_kn, off_kv, off_ku, off_vu, off_acc, off_accm, off_part, off_partm, total;
};

static merc_prefill_layout merc_prefill_get_layout(const ggml_tensor * dst) {
    const ggml_tensor * pk = dst->src[1], * vu = dst->src[6];
    const int r = ggml_get_op_params_i32(dst, 0), rd = ggml_get_op_params_i32(dst, 1), G = ggml_get_op_params_i32(dst, 2);
    const int64_t D = vu->ne[1], Ek = dst->src[0]->ne[0];
    const int64_t n_kv = pk->ne[1] * pk->ne[2];
    merc_prefill_layout L = {};
    L.S          = std::min<int64_t>(n_kv, MERC_PREFILL_SLICE);
    L.CH         = std::min<int64_t>(L.S, MERC_PREFILL_CH);
    L.Ep         = merc_pad16(rd + r);
    L.nkn        = (G - 1) * rd + G * (D - rd);
    L.row_halves = G * Ek + G * D;
    L.nrows      = dst->ne[1] * dst->ne[2];                     // H * T output rows
    const bool sliced = n_kv > L.S;
    size_t off = 0;
    auto take = [&](size_t bytes) { const size_t o = off; off += GGML_PAD(bytes, 256); return o; };
    L.off_kd   = take(sizeof(half)  * ((L.S + 63) / 64 * 64) * L.Ep);
    L.off_rms  = take(sizeof(float) * ((L.S + 63) / 64 * 64) * 4);
    L.off_kn   = take(sizeof(half)  * L.CH * L.nkn);
    L.off_kv   = take(sizeof(half)  * L.S * L.row_halves);
    // up-projections as f16 with the inner dimension zero-padded to the decoded latent's width (multiple of 16):
    // cuBLAS tensor-core GEMMs need aligned leading dimensions, and r (e.g. 579) is odd
    L.kp       = L.Ep - rd;
    L.off_ku   = take(sizeof(half) * L.kp * L.nkn);
    L.off_vu   = take(sizeof(half) * L.kp * D * G);
    L.off_acc  = take(sliced ? sizeof(float)  * L.nrows * D : 0);
    L.off_accm = take(sliced ? sizeof(float2) * L.nrows     : 0);
    L.off_part = take(sliced ? sizeof(float)  * L.nrows * D : 0);
    L.off_partm= take(sliced ? sizeof(float2) * L.nrows     : 0);
    L.total = off;
    return L;
}

size_t ggml_cuda_merc_tq_prefill_get_alloc_size(const ggml_tensor * dst) {
    return GGML_PAD(ggml_nbytes(dst), 256) + merc_prefill_get_layout(dst).total;
}

// merge one slice's partial rows into the running (numerator, max, sum); the last slice also normalizes into dst
template <int D>
static __global__ void merc_prefill_merge(const float * part, const float2 * part_m, float * acc, float2 * acc_m,
                                          float * dst, int nrows) {
    const int row = blockIdx.x, tid = threadIdx.x;
    if (row >= nrows) return;
    const float2 a = acc_m[row], p = part_m[row];
    const float m = fmaxf(a.x, p.x);
    const float sa = expf(a.x - m), sp = expf(p.x - m);
    const float l = sa * a.y + sp * p.y;
    for (int i = tid; i < D; i += blockDim.x) {
        const float v = sa * acc[(int64_t) row * D + i] + sp * part[(int64_t) row * D + i];
        if (dst) dst[(int64_t) row * D + i] = l > 0.0f ? v / l : 0.0f;
        else     acc[(int64_t) row * D + i] = v;
    }
    if (!dst && tid == 0) acc_m[row] = make_float2(m, l);
}

void ggml_cuda_op_merc_tq_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * pk = dst->src[1], * un = dst->src[4], * ku = dst->src[5], * vu = dst->src[6];
    const ggml_tensor * mask = dst->src[7];
    const int r = ggml_get_op_params_i32(dst, 0), rd = ggml_get_op_params_i32(dst, 1), G = ggml_get_op_params_i32(dst, 2);
    const float scale = ggml_get_op_params_f32(dst, 3), freq_base = ggml_get_op_params_f32(dst, 4);
    const int D = (int) vu->ne[1], Ek = (int) q->ne[0], nres = (G - 1) * rd, nope = D - rd;
    const int64_t n_kv = pk->ne[1] * pk->ne[2];
    GGML_ASSERT(Ek == 448 && D == 256 && G == 4 && n_kv % 256 == 0);   // FATTN_KQ_STRIDE
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16);
    const merc_prefill_layout L = merc_prefill_get_layout(dst);
    char * ws = (char *) dst->data + GGML_PAD(ggml_nbytes(dst), 256);
    half   * kd   = (half *)   (ws + L.off_kd);
    float  * rmsd = (float *)  (ws + L.off_rms);
    half   * kn   = (half *)   (ws + L.off_kn);
    half   * kv   = (half *)   (ws + L.off_kv);
    float  * acc  = (float *)  (ws + L.off_acc);
    float2 * accm = (float2 *) (ws + L.off_accm);
    float  * part = (float *)  (ws + L.off_part);
    float2 * partm= (float2 *) (ws + L.off_partm);
    cudaStream_t st = ctx.stream();
    const int64_t nb_cell = pk->ne[1] == 1 ? pk->nb[2] : pk->nb[1];

    half * KU = (half *) (ws + L.off_ku), * VU = (half *) (ws + L.off_vu);
    merc_pad_rows_f16<<<(int) ((L.kp * L.nkn + 255) / 256), 256, 0, st>>>(ku->data, ku->type == GGML_TYPE_F16, r, (int) L.kp, L.nkn, KU);
    merc_pad_rows_f16<<<(int) ((L.kp * D * G + 255) / 256), 256, 0, st>>>(vu->data, vu->type == GGML_TYPE_F16, r, (int) L.kp, (int64_t) D * G, VU);
    const size_t smem_d = sizeof(float) * (rd * rd + 8 * rd + 32) + sizeof(uint32_t) * 64 * pk->ne[0];
    CUDA_CHECK(cudaFuncSetAttribute(merc_tq_decode_f16, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_d));
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), st));
    const float alpha = 1.0f, beta = 0.0f;

    // flash-attention descriptors over the slice buffer: per cell [K_0..K_{G-1} | V_0..V_{G-1}] f16
    const size_t rowb = sizeof(half) * L.row_halves;
    ggml_tensor Kt = {}, Vt = {}, Mt = *mask, fa = *dst;
    Kt.type = GGML_TYPE_F16; Kt.ne[0] = Ek; Kt.ne[2] = G; Kt.ne[3] = 1;
    Kt.nb[0] = sizeof(half); Kt.nb[1] = rowb; Kt.nb[2] = sizeof(half) * Ek;
    Vt.type = GGML_TYPE_F16; Vt.ne[0] = D;  Vt.ne[2] = G; Vt.ne[3] = 1;
    Vt.nb[0] = sizeof(half); Vt.nb[1] = rowb; Vt.nb[2] = sizeof(half) * D;
    Kt.data = kv; Vt.data = kv + (int64_t) G * Ek;
    fa.op = GGML_OP_FLASH_ATTN_EXT;
    memset(fa.op_params, 0, sizeof(fa.op_params));
    ggml_set_op_params_f32(&fa, 0, scale);
    ggml_set_op_params_i32(&fa, 3, GGML_PREC_F32);
    for (int i = 0; i < GGML_MAX_SRC; ++i) fa.src[i] = nullptr;
    fa.src[0] = const_cast<ggml_tensor *>(q); fa.src[1] = &Kt; fa.src[2] = &Vt; fa.src[3] = &Mt;

    const int nslices = (int) ((n_kv + L.S - 1) / L.S);
    for (int s = 0; s < nslices; ++s) {
        const int64_t js = (int64_t) s * L.S;
        const int nc = (int) std::min<int64_t>(L.S, n_kv - js);
        merc_tq_decode_f16<<<(nc + 63) / 64, 256, smem_d, st>>>((const char *) pk->data + js * nb_cell, nb_cell,
            (int) pk->ne[0], (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, un->data,
            un->type == GGML_TYPE_F16, kd, (int) L.Ep, nc, G, r, rd, powf(freq_base, -2.0f / rd), rmsd);
        for (int j0 = 0; j0 < nc; j0 += (int) L.CH) {
            const int ncc = (int) std::min<int64_t>(L.CH, nc - j0);
            const half * C = kd + (int64_t) j0 * L.Ep + rd;       // latent, column-major r x ncc, ld Ep
            // key features (nkn x ncc) = KU^T C into the chunk buffer; values (G*D x ncc) = VU^T C into the slice rows
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int) L.nkn, ncc, (int) L.kp,
                &alpha, KU, CUDA_R_16F, (int) L.kp, C, CUDA_R_16F, (int) L.Ep, &beta, kn, CUDA_R_16F, (int) L.nkn,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, G * D, ncc, (int) L.kp,
                &alpha, VU, CUDA_R_16F, (int) L.kp, C, CUDA_R_16F, (int) L.Ep, &beta, kv + (int64_t) j0 * L.row_halves + G * Ek,
                CUDA_R_16F, (int) L.row_halves, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            merc_tq_expand_assemble<<<ncc, 256, 0, st>>>(kd, (int) L.Ep, rmsd, kn, nullptr, kv, rowb,
                j0, ncc, G, rd, nres, nope, D);
        }
        Kt.ne[1] = nc; Vt.ne[1] = nc;
        Kt.nb[3] = Vt.nb[3] = rowb * nc;
        Mt.data = (char *) mask->data + js * mask->nb[0];
        Mt.ne[0] = nc;
        if (nslices == 1) {
            ggml_cuda_flash_attn_ext_mma_f16_case<448, 256, 16, 4>(ctx, &fa);      // fa.data == dst->data
        } else if (s == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_partial_case<448, 256, 16, 4>(ctx, &fa, acc, accm);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_partial_case<448, 256, 16, 4>(ctx, &fa, part, partm);
            merc_prefill_merge<256><<<(int) L.nrows, 256, 0, st>>>(part, partm, acc, accm,
                s == nslices - 1 ? (float *) dst->data : nullptr, (int) L.nrows);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
