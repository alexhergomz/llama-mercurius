#pragma once
#include "common.cuh"

// Mercurius-1-4B compressed attention cache (TurboQuant-MSE 4-bit): see ggml_merc_tq_pack / ggml_merc_tq_unpack.
void ggml_cuda_op_merc_tq_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_merc_tq_unpack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_merc_tq_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_merc_tq_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
