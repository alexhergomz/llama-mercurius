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
