# Multi-head Latent Attention (MLA)

GQA shrinks the cache by *sharing* K/V across heads ([file 02](02_mqa_and_gqa.md)) — a coarse move, since it throws away per-head resolution wholesale. DeepSeek's MLA (V2, 2024; carried into V3) asks a better question: instead of storing fewer heads, **store a compressed representation and decompress on the fly.** It's the most interesting attention variant of the 2024 crop, and it contains one genuinely subtle problem — an incompatibility with RoPE that forces an unusual fix, and that is the real lesson of this file.

**Config anchor:** DeepSeek-V3, read from its published config — `d_model = 7168`, `L = 61`, `H = 128`, `kv_lora_rank = 512`, `qk_rope_head_dim = 64`, `qk_nope_head_dim = 128`, `v_head_dim = 128`.

## The core idea: cache a latent, not K and V

Instead of projecting the residual stream to full-width keys and values and caching those, MLA projects **down** to a low-rank latent `c_t`, caches *that*, and projects back **up** to per-head K and V inside the attention computation:

```
c_t   = x_t W_DKV                      # down-projection to d_c = 512  ← THIS is what's cached
K_t^h = c_t W_UK^h                     # up-project to each head's key   (128 heads × 128 dims)
V_t^h = c_t W_UV^h                     # up-project to each head's value
```

The cache holds `d_c = 512` numbers per token per layer instead of `2 × n_kv × D_h`. Compare against the alternatives at DeepSeek-V3's actual depth and head count (verified):

| Scheme at `L = 61`, `H = 128`, `D_h = 128` | KV bytes/token |
|---|---|
| MHA (`n_kv = 128`) | 3,904 KiB |
| GQA (`n_kv = 8`) | 244 KiB |
| **MLA** (`d_c = 512` + 64 rope dims) | **68.6 KiB** |

**57× smaller than MHA and 3.6× smaller than GQA-8** — while keeping *128 distinct heads*, each with its own `W_UK`/`W_UV`. That's the pitch: GQA buys cache savings by deleting head diversity; MLA buys them by exploiting the fact that K and V are **low-rank in practice** — you don't need 128 independent 128-dim key spaces, you need one 512-dim subspace that 128 head-specific readouts can project out of.

## The trick that makes it fast, not just small

A skeptic's objection: you've traded memory for compute — every decode step must now up-project the whole cache. If MLA required materializing `K` and `V` per step it would be a bad trade at scale.

It doesn't, because of **matrix absorption**. The attention score is a bilinear form ([1.4/03](../../part1_math_foundations/1.4_optional_deeper_knowledge/03_bilinear_and_quadratic_forms.md)):

```
q^h · K_t^h  =  (x_q W_Q^h) · (c_t W_UK^h)  =  x_q (W_Q^h W_UK^hᵀ) c_tᵀ
                                                    └── fold into ONE matrix, offline ──┘
```

Since only the *product* `W_Q^h W_UK^hᵀ` matters, you precompute it once and apply it to the query. The cached latent `c_t` is then consumed **directly** — never decompressed. Same absorption works on the value side with `W_UV^h` folded into `W_O`. So MLA is not "compress then decompress"; at inference it's "compress once, and rewrite the query/output matrices to speak the latent's language." This is the gauge-freedom observation from [1.4/03](../../part1_math_foundations/1.4_optional_deeper_knowledge/03_bilinear_and_quadratic_forms.md) turned into a systems optimization.

DeepSeek also low-rank-compresses the *queries* (`q_lora_rank = 1536`) — not for cache reasons (queries aren't cached) but to cut activation memory and parameters during training.

## The RoPE problem — and why keys get split in two

Here is the subtlety, and it's worth working through slowly because it's the kind of interaction that only shows up when you compose two good ideas.

RoPE rotates keys by an angle depending on **absolute position** ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)). So the key actually used at position `t` is `R_t · (c_t W_UK)`, and the score involves `x_q W_Q R_{t}... ` — the rotation sits **between** `W_Q` and `W_UK`. That breaks absorption: `W_Q^h R_t W_UK^hᵀ` depends on `t`, so you'd need a *different* folded matrix for every position. Absorption is dead, and you're back to materializing keys per step.

DeepSeek's fix is to **decouple** the two jobs a key does. Each key is split into two concatenated pieces:

```
K_t^h = [  K_nope^h (128 dims)  ‖  K_rope (64 dims)  ]
          from the latent c_t,     from x_t directly,
          NO rotation applied      rotated by RoPE, SHARED across all heads
```

- The **NoPE part** carries content, comes from the compressed latent, and gets no rotation — so absorption works on it.
- The **RoPE part** carries position, is computed straight from `x_t`, is rotated normally, and is **shared across all 128 heads** (MQA-style, one 64-dim vector per token) — so caching it costs almost nothing.

The score is the sum of the two pieces' dot products, which is exactly what concatenation gives you for free. Cache per token per layer is therefore `d_c + qk_rope_head_dim = 512 + 64 = 576` values — the `68.6 KiB/token` above.

Two things to take from this. First, **position and content genuinely are separable jobs**, and MLA exploits that as an engineering resource — the same "addressing vs. payload" split as the QK-vs-OV circuit distinction ([5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)). Second, **RoPE's position-dependence is a real constraint on architectural composition**, not a neutral module you can drop anywhere; expect any future KV-compression scheme to have to answer this same question.

## Cost, and why adoption is limited

MLA's downsides are real and explain why GQA still dominates:

- **Implementation complexity.** Split heads, two RoPE regimes, absorbed matrices at inference but unabsorbed at training, and a non-standard shape everywhere. GQA is ten lines; MLA is a subsystem.
- **Kernel support.** FlashAttention and friends assume standard K/V layouts ([7.2/02](../7.2_efficient_attention/02_flashattention.md)); MLA needed bespoke kernels, which for a long time meant "works well in DeepSeek's stack."
- **More FLOPs at prefill** (the up-projections are real work when you can't amortize them across a long generation), so the win is concentrated in long-context, high-batch decode — exactly DeepSeek's target.

DeepSeek reports MLA *beating* MHA on quality at a fraction of the cache, which if it holds up broadly is the strongest form of the claim: not a tradeoff but a strict improvement, with the low-rank structure acting as useful regularization. Treat that as their measurement on their models rather than a settled field-wide result — but note that Kimi K2 adopted MLA too, so it's no longer a single-lab technique.

## Why it matters in modern LLM work

- **It's the best current example of "compress the cache" as a research direction**, and the numbers to beat.
- **The absorption trick is reusable reasoning:** whenever only a product of learned matrices is observable, you can refactor the factorization for free ([1.4/03](../../part1_math_foundations/1.4_optional_deeper_knowledge/03_bilinear_and_quadratic_forms.md)) — the same insight that makes `W_Q`/`W_K` individually meaningless.
- **The RoPE incompatibility is a case study in composition risk** — two independently-correct designs that interfere, resolved by splitting a tensor along a functional seam.

## Self-check

1. What exactly is stored in an MLA cache, and how big is it per token per layer for DeepSeek-V3?
2. Explain matrix absorption and why it means MLA doesn't pay a decompression cost at decode.
3. Precisely why does RoPE break absorption? Name the offending term.
4. Why is the RoPE part of the key shared across heads while the NoPE part is per-head?
5. MLA is smaller *and* reportedly better than MHA. Why might compression improve quality rather than degrade it?
6. Give two reasons a team might still choose GQA over MLA.

### Answers

1. A single latent vector `c_t` of `kv_lora_rank = 512` values, plus a shared 64-dim RoPE key — **576 values per token per layer** (68.6 KiB/token across 61 layers in bf16). Notably, `V` is not stored at all; it's reconstructed from the same latent.
2. Only the product `W_Q^h W_UK^hᵀ` affects the score, so it can be precomputed into one matrix applied to the query. The cached latent is then dotted against the transformed query directly — keys are never materialized. So the "up-projection" happens once, offline, in weight space, rather than per-token at runtime.
3. RoPE inserts a position-dependent rotation between the two matrices: the score becomes `x_q W_Q^h R_{t} W_UK^hᵀ c_tᵀ`. Because `R_t` varies with `t`, the sandwiched product `W_Q^h R_t W_UK^hᵀ` is a *different matrix at every position*, so it cannot be folded once into weights. `R_t` is the offending term.
4. The RoPE part carries only **position**, which is identical for all heads at a given token — there's nothing head-specific to encode, so one shared 64-dim rotated vector suffices and costs 64 values instead of `128 × 64`. The NoPE part carries **content**, which is exactly what heads must read differently, so it stays per-head (and can, since absorption applies to it).
5. Because `K` and `V` are empirically low-rank — 128 heads do not use 128 genuinely independent key subspaces — so a 512-dim latent may lose little real information while acting as a **bottleneck regularizer**, suppressing directions that were noise. Same argument as low-rank structure being an asset elsewhere ([1.4/01](../../part1_math_foundations/1.4_optional_deeper_knowledge/01_dimension_span_and_rank.md)); the compression is only lossy with respect to capacity the model wasn't productively using.
6. (a) **Complexity/tooling** — GQA is a shape change with universal kernel support; MLA needs custom kernels and has two RoPE paths and train-vs-infer weight refactoring. (b) **Workload fit** — MLA's extra prefill FLOPs and implementation cost pay off mainly at long context with high batch; a short-context or low-volume deployment gets most of the benefit from GQA at a fraction of the engineering risk.

## Exercise

Verify the two central claims numerically. (a) Implement absorption: build random `W_Q ∈ R^(D×D_h)`, `W_UK ∈ R^(d_c×D_h)`, a latent `c`, and a query input `x`; confirm `(x W_Q) · (c W_UK) == x (W_Q W_UKᵀ) cᵀ` to float tolerance, then time both paths for a 4096-token cache to see why absorption matters. (b) Now insert RoPE on the key and confirm the identity **breaks** — and that it breaks *differently at every position*, which is the precise statement of self-check 3. (c) Implement the decoupled fix: split the key into a 128-dim NoPE half from the latent and a 64-dim shared RoPE half from `x`, and confirm the score equals the sum of two dot products while absorption still holds on the NoPE half. (d) Tabulate cache bytes/token for MHA, GQA-8, and MLA at DeepSeek-V3's config and reproduce the 57× and 3.6× ratios.
