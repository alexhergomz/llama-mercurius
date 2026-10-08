// Mercurius-1-4B (Minerva Laboratories): Qwen3.5-4B rebuilt with Gated DeltaNet-2 linear attention (channel-wise
// decay, erase and write gates), absorbed MLA with a decoupled RoPE key, and a TurboQuant-MSE 4-bit KV cache.
//
// GDN-2:  S <- D S + k (w*v - S^T (b*k))^T, D = diag(exp(g)), g = -exp(A_log) * softplus(a + dt_bias)
//         a, b, w come from factored gates: tiled base rows (one per value head, repeated per channel) + VeRA
//         (diag(vb) B diag(vd) A x, A and B shared by every gate), plus a rank-32 LoRA on the decay.
// MLA:    the converter (OpenDecider deploy/mercurius_gguf.py) folds the post-norm query map, the RoRoPE frequency
//         mixing, the key up-projections and the latent rotation into one per-head matrix:
//             [q_rope | q_abs] = Q_MAP_h RMSNorm(q_h)
//         scores against the cache [rope(R0^T kr) | c] divided by the per-group key rms; attention runs on the latent
//         and V is up-projected per group: o_h = V_UP_g (sum_s p_s c_s), then the sigmoid output gate.
// Cache:  one packed row of TurboQuant codes per token and attention layer (ggml_merc_tq_pack / _unpack), no V cache.

#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_mercurius::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
    }

    ml.get_key_or_arr(LLM_KV_ATTENTION_LATENT_RANKS, hparams.merc_latent_rank, hparams.n_layer_all);
    ml.get_key_or_arr(LLM_KV_ATTENTION_CACHE_WORDS,  hparams.merc_cache_words, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_K_NORM_EPS, hparams.merc_k_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_SCALE,      hparams.f_attention_scale);
    ml.get_key(LLM_KV_VERA_RANK,            hparams.merc_vera_rank);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        GGML_ASSERT((hparams.merc_cache_words[i] > 0) == !hparams.is_recr(i));
    }

    type = hparams.n_layer() == 32 && hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_UNKNOWN;
}

void llama_model_mercurius::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim = hparams.ssm_d_inner / n_v_heads;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t gate_out   = head_k_dim * n_v_heads;      // channel-wise gates: per value head, per key channel
    const int64_t vera_rank  = hparams.merc_vera_rank;

    merc_vera_a = create_tensor(tn(LLM_TENSOR_VERA_A, "weight"), { n_embd, vera_rank }, 0);
    merc_vera_b = create_tensor(tn(LLM_TENSOR_VERA_B, "weight"), { vera_rank, gate_out }, 0);

    // n_rot: from LLAMA_LOAD_LOCALS
    const int64_t D     = hparams.n_embd_head_k();

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            const int64_t r = hparams.merc_latent_rank[il];
            const int64_t G = hparams.n_head_kv(il);
            layer.wq          = create_tensor(tn(LLM_TENSOR_ATTN_Q,      "weight", il), { n_embd, D * n_head * 2 }, 0);
            layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { D * n_head, n_embd }, 0);
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { D }, 0);
            layer.merc_q_map        = create_tensor(tn(LLM_TENSOR_ATTN_Q_MAP,        "weight", il), { D, n_rot + r, n_head }, 0);
            layer.merc_k_rope       = create_tensor(tn(LLM_TENSOR_ATTN_K_ROPE,       "weight", il), { n_embd, n_rot }, 0);
            layer.merc_rope_unrot   = create_tensor(tn(LLM_TENSOR_ATTN_ROPE_UNROT,   "weight", il), { n_rot, n_rot }, 0);
            layer.merc_latent       = create_tensor(tn(LLM_TENSOR_ATTN_LATENT,       "weight", il), { n_embd, r }, 0);
            layer.merc_v_up         = create_tensor(tn(LLM_TENSOR_ATTN_V_UP,         "weight", il), { r, D, G }, 0);
            layer.merc_k_rms        = create_tensor(tn(LLM_TENSOR_ATTN_K_RMS,        "weight", il), { n_embd, D * G }, 0);
            layer.merc_tq_latent_cb = create_tensor(tn(LLM_TENSOR_ATTN_TQ_LATENT_CB, "weight", il), { 31 }, 0);
            layer.merc_tq_rope_cb   = create_tensor(tn(LLM_TENSOR_ATTN_TQ_ROPE_CB,   "weight", il), { 31 }, 0);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, key_dim * 2 + value_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { gate_out }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { gate_out }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, 0);
            layer.merc_a_lora_a = create_tensor(tn(LLM_TENSOR_SSM_A_LORA_A, "weight", il), { n_embd, 32 }, TENSOR_NOT_REQUIRED);
            if (layer.merc_a_lora_a) {
                const int64_t lr = layer.merc_a_lora_a->ne[1];
                layer.merc_a_lora_b = create_tensor(tn(LLM_TENSOR_SSM_A_LORA_B, "weight", il), { lr, gate_out }, 0);
            }
            const llm_tensor base[3] = { LLM_TENSOR_GATE_A_BASE, LLM_TENSOR_GATE_BE_BASE, LLM_TENSOR_GATE_BW_BASE };
            const llm_tensor vd[3]   = { LLM_TENSOR_GATE_A_VD,   LLM_TENSOR_GATE_BE_VD,   LLM_TENSOR_GATE_BW_VD };
            const llm_tensor vb[3]   = { LLM_TENSOR_GATE_A_VB,   LLM_TENSOR_GATE_BE_VB,   LLM_TENSOR_GATE_BW_VB };
            for (int g = 0; g < 3; ++g) {
                layer.merc_gate_base[g] = create_tensor(tn(base[g], "weight", il), { n_embd, n_v_heads }, 0);
                layer.merc_gate_vd[g]   = create_tensor(tn(vd[g],   "weight", il), { vera_rank }, 0);
                layer.merc_gate_vb[g]   = create_tensor(tn(vb[g],   "weight", il), { gate_out }, 0);
            }
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff,   n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mercurius::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mercurius::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_ffn(cur,
            model.layers[il].ffn_up,   NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// one factored gate: tiled base (one row per value head, repeated per key channel) + VeRA [+ decay LoRA]
ggml_tensor * llama_model_mercurius::graph::build_gate(ggml_tensor * cur, ggml_tensor * vera_u, int g, int il) {
    const auto & layer = model.layers[il];
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_tokens   = cur->ne[1];

    ggml_tensor * base = ggml_mul_mat(ctx0, layer.merc_gate_base[g], cur);                 // [n_v, T]
    base = ggml_reshape_3d(ctx0, base, 1, n_v_heads, n_tokens);
    base = ggml_repeat_4d(ctx0, base, head_k_dim, n_v_heads, n_tokens, 1);                // [dk, n_v, T]
    base = ggml_reshape_2d(ctx0, base, head_k_dim * n_v_heads, n_tokens);

    ggml_tensor * t = ggml_mul(ctx0, vera_u, layer.merc_gate_vd[g]);                      // diag(vd) A x
    t = ggml_mul_mat(ctx0, model.merc_vera_b, t);                                         // B (.)
    t = ggml_mul(ctx0, t, layer.merc_gate_vb[g]);                                         // diag(vb)
    ggml_tensor * out = ggml_add(ctx0, base, t);

    if (g == 0 && layer.merc_a_lora_a) {
        out = ggml_add(ctx0, out, ggml_mul_mat(ctx0, layer.merc_a_lora_b, ggml_mul_mat(ctx0, layer.merc_a_lora_a, cur)));
    }
    return out;                                                                           // [dk * n_v, T]
}

ggml_tensor * llama_model_mercurius::graph::build_layer_attn_linear(llm_graph_input_rs * inp, ggml_tensor * cur, int il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_inner / num_v_heads;
    GGML_ASSERT(n_seqs != 0 && ubatch.equal_seqs() && ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx0, layer.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    ggml_tensor * z = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);

    // gates
    ggml_tensor * vera_u = ggml_mul_mat(ctx0, model.merc_vera_a, cur);                     // A x, shared
    ggml_tensor * a  = build_gate(cur, vera_u, 0, il);
    ggml_tensor * be = ggml_sigmoid(ctx0, build_gate(cur, vera_u, 1, il));                // erase gate b
    ggml_tensor * bw = ggml_sigmoid(ctx0, build_gate(cur, vera_u, 2, il));                // write gate w
    cb(be, "gdn2_erase", il);
    cb(bw, "gdn2_write", il);

    ggml_tensor * gdec = ggml_softplus(ctx0, ggml_add(ctx0, a, layer.ssm_dt));
    gdec = ggml_mul(ctx0, gdec, layer.ssm_a);                                              // -exp(A_log) * softplus
    gdec = ggml_reshape_4d(ctx0, gdec, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);    // KDA: [S, H_v, T, B]
    be   = ggml_reshape_4d(ctx0, be,   head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    bw   = ggml_reshape_4d(ctx0, bw,   head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    cb(gdec, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel     = layer.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = hparams.ssm_d_inner + 2 * num_k_heads * head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const size_t  nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
        head_k_dim * num_k_heads * ggml_element_size(conv_out));
    ggml_tensor * v = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
        ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    q = ggml_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
    k = ggml_l2_norm(ctx0, k, hparams.f_norm_rms_eps);
    v = ggml_mul(ctx0, v, bw);                                                             // write gate folded into v

    // the fused op is the one that takes a channel-wise beta (the chunked graph assumes a scalar)
    ggml_tensor * output;
    if (cparams.n_rs_seq == 0) {
        auto attn_out = build_delta_net_fused(q, k, v, gdec, be, state, il);
        output = attn_out.first;
        const auto kv_head = mctx_cur->get_head();
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, attn_out.second,
                ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));
    } else {
        output = build_recurrent_attn(inp, ssm_states_all, q, k, v, gdec, be, state, il);
    }

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * normed = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    normed = ggml_mul(ctx0, normed, ggml_silu(ctx0, z_2d));
    ggml_tensor * flat = ggml_reshape_3d(ctx0, normed, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    cur = ggml_mul_mat(ctx0, layer.ssm_out, flat);
    cb(cur, "linear_attn_out", il);
    return ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
}

ggml_tensor * llama_model_mercurius::graph::build_layer_attn(llm_graph_input_attn_kv * inp, ggml_tensor * cur,
                                                             ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    const int64_t D   = hparams.n_embd_head_k();
    const int64_t H   = n_head;
    const int64_t G   = hparams.n_head_kv(il);
    const int64_t rep = H / G;
    const int64_t rd  = n_rot;
    const int64_t r   = hparams.merc_latent_rank[il];
    const int64_t T   = cur->ne[1];

    // query + output gate, per head
    ggml_tensor * qg = ggml_mul_mat(ctx0, layer.wq, cur);                                  // [2D*H, T]
    ggml_tensor * q = ggml_view_3d(ctx0, qg, D, H, T, ggml_element_size(qg) * 2 * D, ggml_element_size(qg) * 2 * D * H, 0);
    ggml_tensor * gate = ggml_view_3d(ctx0, qg, D, H, T, ggml_element_size(qg) * 2 * D, ggml_element_size(qg) * 2 * D * H,
                                      ggml_element_size(qg) * D);
    gate = ggml_cont_2d(ctx0, gate, D * H, T);
    q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);                      // [D, H, T]

    // [q_rope | q_abs] = Q_MAP_h y_h   (batched over heads)
    ggml_tensor * qh = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));                 // [D, T, H]
    ggml_tensor * qq = ggml_mul_mat(ctx0, layer.merc_q_map, qh);                           // [rd + r, T, H]
    ggml_tensor * q_rope = ggml_view_3d(ctx0, qq, rd, T, H, qq->nb[1], qq->nb[2], 0);
    ggml_tensor * q_abs  = ggml_view_3d(ctx0, qq, r,  T, H, qq->nb[1], qq->nb[2], rd * ggml_element_size(qq));
    q_rope = ggml_cont(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3));                      // [rd, H, T] for rope
    q_rope = ggml_rope_ext(ctx0, q_rope, inp_pos, nullptr, rd, LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                           ext_factor, attn_factor, beta_fast, beta_slow);
    q_rope = ggml_cont(ctx0, ggml_permute(ctx0, q_rope, 0, 2, 1, 3));                      // [rd, T, H]
    ggml_tensor * q_full = ggml_concat(ctx0, q_rope, ggml_cont(ctx0, q_abs), 0);          // [rd + r, T, H]

    // this step's cache entries: RoPE key and latent (rotations folded), per-group key rms
    ggml_tensor * kr = ggml_mul_mat(ctx0, layer.merc_k_rope, cur);                         // [rd, T]
    ggml_tensor * c  = ggml_mul_mat(ctx0, layer.merc_latent, cur);                         // [r, T]
    ggml_tensor * kk = ggml_mul_mat(ctx0, layer.merc_k_rms, cur);                          // [D*G, T]
    kk = ggml_reshape_3d(ctx0, kk, D, G, T);
    ggml_tensor * rms = ggml_sum_rows(ctx0, ggml_sqr(ctx0, kk));                           // [1, G, T]
    rms = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, rms, 1.0f / D, hparams.merc_k_norm_eps));
    rms = ggml_reshape_2d(ctx0, rms, G, T);

    const int words = hparams.merc_cache_words[il];
    ggml_tensor * packed = ggml_merc_tq_pack(ctx0, c, kr, rms, inp_pos, layer.merc_tq_latent_cb, layer.merc_tq_rope_cb, words);
    ggml_build_forward_expand(gf, q_full);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, ggml_reshape_3d(ctx0, packed, words, 1, T), inp->get_k_idxs(), il));

    // decode the layer's cache (past + this step)
    ggml_tensor * kc = mctx_cur->get_k(ctx0, il);                                           // [words, 1, n_kv, ns]
    GGML_ASSERT(kc->ne[3] == 1 && "mercurius: one KV stream");
    const int64_t n_kv = kc->ne[2];
    ggml_tensor * dec = ggml_merc_tq_unpack(ctx0, kc, layer.merc_tq_latent_cb, layer.merc_tq_rope_cb, r, rd, G, 0);
    ggml_tensor * pos = ggml_merc_tq_unpack(ctx0, kc, layer.merc_tq_latent_cb, layer.merc_tq_rope_cb, r, rd, G, 1);
    ggml_tensor * c_all  = ggml_view_2d(ctx0, dec, r,  n_kv, dec->nb[1], 0);
    ggml_tensor * kt_all = ggml_view_2d(ctx0, dec, rd, n_kv, dec->nb[1], r * sizeof(float));
    ggml_tensor * rms_all = ggml_view_2d(ctx0, dec, G, n_kv, dec->nb[1], (r + rd) * sizeof(float));

    ggml_tensor * kt = ggml_mul_mat(ctx0, layer.merc_rope_unrot, kt_all);                  // R0^T, [rd, n_kv]
    kt = ggml_reshape_3d(ctx0, kt, rd, 1, n_kv);
    kt = ggml_rope_ext(ctx0, kt, pos, nullptr, rd, LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                       ext_factor, attn_factor, beta_fast, beta_slow);
    ggml_tensor * k_full = ggml_concat(ctx0, ggml_reshape_2d(ctx0, kt, rd, n_kv), ggml_cont(ctx0, c_all), 0);   // [rd + r, n_kv]

    ggml_tensor * kq = ggml_mul_mat(ctx0, k_full, q_full);                                 // [n_kv, T, H]
    kq = ggml_reshape_4d(ctx0, kq, n_kv, T, rep, G);                                       // head h = j + rep*g
    ggml_tensor * rms_t = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, rms_all)), n_kv, 1, 1, G);
    // unwritten cells of the KV window decode to rms 0: keep their (masked) scores finite instead of 0/0
    rms_t = ggml_clamp(ctx0, rms_t, 1e-6f, INFINITY);
    kq = ggml_div(ctx0, kq, rms_t);                                                        // per-group key normaliser
    kq = ggml_reshape_3d(ctx0, kq, n_kv, T, H);
    kq = ggml_soft_max_ext(ctx0, kq, inp->get_kq_mask(), hparams.f_attention_scale, 0.0f);
    cb(kq, "kq_soft_max", il);

    ggml_tensor * c_t = ggml_cont(ctx0, ggml_transpose(ctx0, c_all));                      // [n_kv, r]
    ggml_tensor * o_lat = ggml_mul_mat(ctx0, c_t, kq);                                     // [r, T, H]
    o_lat = ggml_reshape_4d(ctx0, o_lat, r, T, rep, G);
    ggml_tensor * wv = ggml_reshape_4d(ctx0, layer.merc_v_up, r, D, 1, G);
    ggml_tensor * o = ggml_mul_mat(ctx0, wv, o_lat);                                       // [D, T, rep, G]
    o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 3, 1, 2));                                // [D, rep, G, T]
    o = ggml_reshape_2d(ctx0, o, D * H, T);

    o = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, gate));
    cur = ggml_mul_mat(ctx0, layer.wo, o);
    cb(cur, "attn_output", il);
    return cur;
}
