#pragma once
// code-predictor-forward.h: run the 5-layer Qwen3 code predictor over a
// growing context to produce the 15 acoustic codes of one audio frame,
// KV cached.
//
// Input:
//   talker_hidden_last [hidden] f32   -- last position hidden state from
//                                        the Talker forward (post final norm)
//   c0                                -- semantic code sampled from the
//                                        Talker codec_head (codebook 0)
// Output:
//   codes[16] = [c0, c1, ..., c15]    -- the full set of codes for one
//                                        frame, ready for decode through
//                                        the codec
//
// The predictor cache is local to a single frame: we reset it at every
// frame, prefill the first two positions (talker_hidden + embed(c0)),
// then decode 14 single-token steps. Total work drops from
// O(sum_{g=0..14} (g+2)^2) = O(1496 token-steps) to O(16) per frame,
// roughly 90x for the inner loop.
//
// Architecture mirrors the Talker block, only differences are:
//   - 5 layers instead of 28
//   - plain 1D RoPE (no multimodal sections)
//   - one private embedding table and one private linear head per
//     acoustic codebook (1..15)
//
// Graph metadata lives in two caller owned persistent arenas, one for
// the T=2 prefill and one for the T=1 steps: each shape class keeps a
// stable first node address so the CUDA graph cache replays instead of
// reinstantiating when the two alternate within a frame.

#include "code-predictor-weights.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "graph-arena.h"
#include "kv-cache.h"
#include "qt-error.h"
#include "sampling.h"
#include "talker-weights.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct CodePredictorOutput {
    // Sixteen codes: c0 from the talker plus c1..c15 from the predictor.
    std::vector<int32_t> codes;
};

// Manual F32 attention chain for the code predictor block. Same shape
// contract as talker_attn_f32: q [hd, T, n_q_heads], k/v [hd, T_full,
// n_kv], output [hd, n_q_heads, T]. Used when use_flash_attn is false.
static struct ggml_tensor * code_predictor_attn_f32(struct ggml_context * ctx,
                                                    struct ggml_tensor *  q,
                                                    struct ggml_tensor *  k,
                                                    struct ggml_tensor *  v,
                                                    struct ggml_tensor *  mask,
                                                    float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Node budget for one predictor graph, same accounting as the talker.
static int code_predictor_graph_max_nodes(int n_layers) {
    return 48 * n_layers + 64;
}

// One Qwen3 decoder block, KV cached. K and V for the T fresh positions
// are written into the cache at [n_past, n_past+T) on dim 1; the
// attention reads the fixed [0, n_kv_pad) window with the mask carrying
// neg inf beyond n_past+T. Returns the layer output [hidden, T].
// use_flash_attn and clamp_fp16 follow the same contract as in
// talker-forward.h.
static struct ggml_tensor * code_predictor_layer_forward(struct ggml_context *        ctx,
                                                         const CodePredictorWeights * cw,
                                                         const TalkerLayer &          layer,
                                                         struct ggml_tensor *         x,
                                                         struct ggml_tensor *         positions,
                                                         struct ggml_tensor *         mask,
                                                         struct ggml_tensor *         k_cache,
                                                         struct ggml_tensor *         v_cache,
                                                         int                          n_past,
                                                         int                          T,
                                                         int                          n_kv_pad,
                                                         bool                         use_flash_attn,
                                                         bool                         clamp_fp16,
                                                         struct ggml_cgraph *         gf) {
    const int   n_q_heads = cw->num_attention_heads;
    const int   n_kv      = cw->num_key_value_heads;
    const int   hd        = cw->head_dim;
    const float eps       = cw->rms_norm_eps;

    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, T);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, T);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, T);

    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Write the fresh positions into the cache.
    struct ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, T, n_kv]
    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    size_t k_off = (size_t) n_past * k_cache->nb[1];
    size_t v_off = (size_t) n_past * v_cache->nb[1];

    struct ggml_tensor * k_dst = ggml_view_3d(ctx, k_cache, hd, T, n_kv, k_cache->nb[1], k_cache->nb[2], k_off);
    struct ggml_tensor * v_dst = ggml_view_3d(ctx, v_cache, hd, T, n_kv, v_cache->nb[1], v_cache->nb[2], v_off);

    struct ggml_tensor * k_cpy = ggml_cpy(ctx, k_perm, k_dst);
    struct ggml_tensor * v_cpy = ggml_cpy(ctx, v_perm, v_dst);
    ggml_build_forward_expand(gf, k_cpy);
    ggml_build_forward_expand(gf, v_cpy);

    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache, hd, n_kv_pad, n_kv, k_cache->nb[1], k_cache->nb[2], 0);
    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache, hd, n_kv_pad, n_kv, v_cache->nb[1], v_cache->nb[2], 0);

    // Q permute [hd, n_q_heads, T] -> [hd, T, n_q_heads] for flash_attn_ext.
    struct ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);

    // Clamp V before attention when clamp_fp16 is set, same rationale
    // as the talker block: sub Ampere CUDA tensor cores accumulate in
    // FP16 and a V projection overflow corrupts everything downstream.
    if (clamp_fp16) {
        v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
    }

    // Attention: fused flash kernel or manual F32 chain. Matches the
    // working acestep qw3lm_build_attn pattern on the fused branch and
    // the omnivoice qwen3_attn_f32 helper on the manual one.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = code_predictor_attn_f32(ctx, q_p, k_full, v_full, mask, scale);
    }

    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, T);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);
    x                      = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    gate                      = ggml_silu(ctx, gate);
    struct ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

// Run one predictor pass: feed `T` fresh embeddings starting at cache
// position `n_past`, run all 5 layers, and pull the logits for the last
// position through lm_head[g_head]. The cache is written as a side
// effect so subsequent decode steps can append a single token.
// use_flash_attn / clamp_fp16 are forwarded as is to every layer.
static bool code_predictor_run(const CodePredictorWeights * cw,
                               KVCache *                    kv,
                               ggml_backend_sched_t         sched,
                               GraphArena *                 arena,
                               struct ggml_tensor *         embd_table,
                               const float *                hidden_row,
                               int32_t                      code_id,
                               int                          T,
                               int                          n_past,
                               int                          talker_hidden,
                               int                          g_head,
                               bool                         use_flash_attn,
                               bool                         clamp_fp16,
                               std::vector<float> *         logits_out) {
    const int vocab    = cw->vocab_size;
    const int n_layers = cw->num_hidden_layers;
    const int T_full   = n_past + T;

    // The attention window spans the whole frame cache (16 slots): a
    // constant width keeps prefill and step graph shapes fixed across
    // frames so the CUDA graph cache replays each of the two flavors.
    const int n_kv_pad = kv->max_seq_len;

    const int             max_nodes = code_predictor_graph_max_nodes(n_layers);
    struct ggml_context * gctx      = graph_arena_begin(arena);

    // Inputs: one code id gathered in graph from embd_table, positions,
    // attention mask, plus the raw talker hidden row on the prefill
    // path (T == 2, hidden_row non NULL) where the sequence is
    // [talker_hidden, embed(c0)]. Steps (T == 1) are pure gathers: the
    // only per step upload is 4 bytes of code id.
    struct ggml_tensor * ids_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, n_kv_pad, T);
    ggml_set_name(ids_in, "sub_code_id");
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");
    ggml_set_input(ids_in);

    struct ggml_tensor * x_in   = ggml_get_rows(gctx, embd_table, ids_in);
    struct ggml_tensor * hid_in = NULL;
    if (T == 2) {
        hid_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, talker_hidden, 1);
        ggml_set_name(hid_in, "talker_hidden_row");
        ggml_set_input(hid_in);
        x_in = ggml_concat(gctx, hid_in, x_in, 1);
    }
    ggml_set_name(x_in, "sub_input");

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, max_nodes, false);

    // small_to_mtp projection: Linear(talker_hidden -> hidden) with bias.
    // When absent (Identity case) the input is already at predictor hidden.
    struct ggml_tensor * h = x_in;
    if (cw->mtp_proj_w) {
        h = ggml_mul_mat(gctx, cw->mtp_proj_w, h);
        if (cw->mtp_proj_b) {
            h = ggml_add(gctx, h, cw->mtp_proj_b);
        }
        ggml_set_name(h, "mtp_proj_out");
    }

    for (int l = 0; l < n_layers; l++) {
        h = code_predictor_layer_forward(gctx, cw, cw->layers[(size_t) l], h, pos_in, mask_in, kv->k[(size_t) l],
                                         kv->v[(size_t) l], n_past, T, n_kv_pad, use_flash_attn, clamp_fp16, gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, cw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, cw->norm_w);

    struct ggml_tensor * logits = ggml_mul_mat(gctx, cw->lm_head[(size_t) g_head], h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "[CodePredictor] FATAL: graph allocation failed\n");
        ggml_backend_sched_reset(sched);
        return false;
    }

    ggml_backend_tensor_set(ids_in, &code_id, 0, sizeof(int32_t));
    if (hid_in) {
        ggml_backend_tensor_set(hid_in, hidden_row, 0, (size_t) talker_hidden * sizeof(float));
    }

    {
        std::vector<int32_t> pos((size_t) T);
        for (int i = 0; i < T; i++) {
            pos[(size_t) i] = n_past + i;
        }
        ggml_backend_tensor_set(pos_in, pos.data(), 0, (size_t) T * sizeof(int32_t));
    }

    {
        std::vector<ggml_fp16_t> mask((size_t) T * (size_t) n_kv_pad);
        const ggml_fp16_t        zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); i++) {
            mask[i] = neg_inf;
        }
        for (int q = 0; q < T; q++) {
            const int q_pos = n_past + q;
            for (int k = 0; k <= q_pos; k++) {
                mask[(size_t) q * (size_t) n_kv_pad + (size_t) k] = zero;
            }
        }
        ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[CodePredictor] FATAL: graph compute failed\n");
        ggml_backend_sched_reset(sched);
        return false;
    }

    logits_out->resize((size_t) vocab);
    size_t row_bytes = (size_t) vocab * sizeof(float);
    ggml_backend_tensor_get(logits, logits_out->data(), (size_t) (T - 1) * row_bytes, row_bytes);

    kv->cur_len = T_full;
    return true;
}

// Run the predictor for one audio frame. Caller passes the talker hidden
// state for the current frame and the already-sampled c0. Sampling
// parameters control greedy (temperature <= 0) vs stochastic. subseq_base
// is the Philox subsequence of the c0 sample for this step; the 15
// acoustic samples consume subseq_base + 1 .. subseq_base + 15.
// Returns the full vector of 16 codes. dump_dir may be NULL.
// use_flash_attn / clamp_fp16 are forwarded as is to every internal run.
static bool code_predictor_step(const TalkerWeights *        tw,
                                const CodePredictorWeights * cw,
                                KVCache *                    kv,
                                ggml_backend_sched_t         sched,
                                GraphArena *                 arena_prefill,
                                GraphArena *                 arena_step,
                                const float *                talker_hidden_last,
                                int                          c0,
                                float                        temperature,
                                int                          top_k,
                                float                        top_p,
                                int64_t                      seed,
                                int64_t                      subseq_base,
                                bool                         use_flash_attn,
                                bool                         clamp_fp16,
                                const char *                 dump_dir,
                                CodePredictorOutput *        out) {
    // sub_input slots live at the talker hidden dimension: both the
    // talker last hidden and the codec_embedding rows feeding the sub
    // network are talker sized in the upstream checkpoint. The graph's
    // mtp_proj brings them down to predictor hidden when present.
    const int talker_hidden = tw->hidden_size;
    const int n_acoustic    = cw->num_acoustic_codebooks;

    if (n_acoustic + 1 > kv->max_seq_len) {
        fprintf(stderr, "[CodePredictor] FATAL: frame width %d exceeds cache max_seq_len %d\n", n_acoustic + 1,
                kv->max_seq_len);
        return false;
    }

    out->codes.assign((size_t) (n_acoustic + 1), 0);
    out->codes[0] = c0;

    // Prefill: two positions, [talker_hidden_last, embed_talker(c0)].
    // The hidden row uploads raw, c0 gathers in graph from the talker
    // codec embedding table.
    kv_cache_reset(kv);

    std::vector<float> logits;
    if (!code_predictor_run(cw, kv, sched, arena_prefill, tw->codec_embedding, talker_hidden_last, c0, 2, 0,
                            talker_hidden, 0, use_flash_attn, clamp_fp16, &logits)) {
        return false;
    }
    {
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1, &u_g);
        if (subseq_base + 1 < 32) {
            fprintf(stderr, "[Sample-CP] g=0 c=%d u=%.10f subseq=%lld\n", cg, (double) u_g,
                    (long long) (subseq_base + 1));
        }
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=0\n");
            return false;
        }
        out->codes[1] = cg;
    }

    // Decode loop: 14 single-token steps. At step g (g=1..14) we feed
    // the id of the code we just sampled, gathered in graph from the
    // group's private embedding table, and read lm_head[g].
    for (int g = 1; g < n_acoustic; g++) {
        if (!code_predictor_run(cw, kv, sched, arena_step, cw->codec_embedding[(size_t) (g - 1)], NULL,
                                out->codes[(size_t) g], 1, kv->cur_len, talker_hidden, g, use_flash_attn, clamp_fp16,
                                &logits)) {
            return false;
        }
        float u_g = 0.0f;
        int   cg = sample_top_k_p(logits.data(), (int) logits.size(), temperature, top_k, top_p, 1.0f, nullptr, 0, seed,
                                  subseq_base + 1 + g, &u_g);
        if (subseq_base + 1 + g < 32) {
            fprintf(stderr, "[Sample-CP] g=%d c=%d u=%.10f subseq=%lld\n", g, cg, (double) u_g,
                    (long long) (subseq_base + 1 + g));
        }
        if (cg < 0) {
            fprintf(stderr, "[CodePredictor] FATAL: sample returned no candidate at g=%d\n", g);
            return false;
        }
        out->codes[(size_t) (g + 1)] = cg;
    }

    if (dump_dir) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        std::vector<int32_t> codes32(out->codes.begin(), out->codes.end());
        int                  n = (int) codes32.size();
        debug_dump_i32_as_f32(&d, "codes-step0", codes32.data(), &n, 1);
    }

    return true;
}
