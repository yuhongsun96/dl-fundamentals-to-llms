# Modern LLM Architecture (Decoder-Only)

A single-page map of what a contemporary large-scale language model actually looks like, end to end, with every standard "trick" in place. This is the target the rest of the curriculum is building toward — many of the components here get their own dedicated treatment in later parts (flagged **(Part N)**), but seeing them assembled once gives the skeleton to hang everything on.

The reference architecture is a **decoder-only Transformer in the LLaMA / GPT lineage** — the design that won the scaling race (Part 6.1). Numbers throughout use an **8B-class** config: `D = 4096`, `L = 32` layers, `H = 32` query heads, `D_h = 128`, `n_kv = 8` KV-head groups (GQA), `d_ff = 14336` (SwiGLU), `V = 128000`, max `S = 8192`.

For scale, the same family at larger sizes (LLaMA-3-style, approximate):

| Class | `D` | `L` | `H` | `D_h` | `n_kv` | `d_ff` | `V` |
|---|---|---|---|---|---|---|---|
| **8B** (reference) | 4096 | 32 | 32 | 128 | 8 | 14336 | 128K |
| **70B** | 8192 | 80 | 64 | 128 | 8 | 28672 | 128K |
| **405B** | 16384 | 126 | 128 | 128 | 8 | 53248 | 128K |

The knobs that move are **width `D`** and **depth `L`** (and `H = D/D_h` follows from width). Notice what *stays fixed*: `D_h = 128` (per-head dim is a hardware/numerics sweet spot, so you add heads, not bigger heads), `n_kv = 8` (KV groups held constant to cap the inference KV cache even as the model grows), `d_ff ≈ 3.5·D` (the SwiGLU ratio), and `V` (vocabulary is a tokenizer choice, independent of model size — so embeddings are a *shrinking* fraction of the total as `D, L` grow: ~13% at 8B, ~3% at 70B, ~1% at 405B).

**Convention:** row-vector (`Y = X W`), the repo default — activations are rows, so `(B, S, D) @ (D, d_out) → (B, S, d_out)`. Dimension names follow [NOTATION.md](NOTATION.md).

---

## The big picture

```
                        token ids   (B, S)
                            │
                  ┌─────────▼──────────┐
                  │  Token Embedding   │   W_E ∈ R^(V × D)   — pure lookup
                  └─────────┬──────────┘
                            │  x₀  (B, S, D)   ← the residual stream is born
                            │
            ╔═══════════════▼════════════════╗
            ║      Decoder Block   × L        ║   ← detail below
            ╚═══════════════┬════════════════╝
                            │  x_L  (B, S, D)
                  ┌─────────▼──────────┐
                  │   Final RMSNorm    │   γ_f ∈ R^D
                  └─────────┬──────────┘
                  ┌─────────▼──────────┐
                  │    Unembedding     │   W_U ∈ R^(D × V)   (often = W_Eᵀ, "tied")
                  └─────────┬──────────┘
                            │  logits  (B, S, V)
                            ▼
                  softmax  →  next-token distribution
```

The model is: embed tokens into the residual stream, push the stream through `L` identical blocks, normalize once more, project back to vocabulary logits. No positional embedding is added at the input — modern models inject position **inside attention** via RoPE (see the block).

## One decoder block, zoomed in

Pre-norm, with the **residual stream as an un-normalized highway** down the left. Each sublayer reads a *normalized copy* of the stream and adds its result back to the raw stream.

```
   x_in (B,S,D) ──────────────────────────────┐  residual highway
        │                                      │  (never normalized)
   ┌────▼─────┐                                │
   │ RMSNorm  │  γ₁ ∈ R^D                       │
   └────┬─────┘                                │
   ┌────▼───────────────────────────────────┐  │
   │  Grouped-Query Self-Attention           │  │
   │    Q = x̂ W_Q     (H heads)              │  │
   │    K = x̂ W_K, V = x̂ W_V  (n_kv groups) │  │   ← GQA: fewer K/V than Q
   │    q, k ← RoPE(q, k, pos)               │  │   ← position injected here
   │    q, k ← QK-norm(q, k)        (optional)│  │   ← caps logit growth
   │    A = softmax( Q Kᵀ / √D_h + causal )  │  │   ← (B,H,S,S), masked
   │    attn = (A V) W_O                     │  │
   └────┬────────────────────────────────────┘  │
        │ attn                                   │
       (+)◄─────────────────────────────────────┘   x_mid = x_in + attn
        │
        ├────────────────────────────────────────┐  residual highway
   ┌────▼─────┐                                   │
   │ RMSNorm  │  γ₂ ∈ R^D                          │
   └────┬─────┘                                   │
   ┌────▼────────────────────────────────────┐    │
   │  SwiGLU Feed-Forward                      │   │
   │    gate = x̂ W_gate   (B,S,d_ff)           │   │
   │    up   = x̂ W_up     (B,S,d_ff)           │   │
   │    h    = SiLU(gate) ⊙ up                 │   │   ← gated activation
   │    ffn  = h W_down   (B,S,D)              │   │
   └────┬─────────────────────────────────────┘    │
        │ ffn                                       │
       (+)◄────────────────────────────────────────┘   x_out = x_mid + ffn
        │
        ▼  to next block
```

---

## Component walkthrough

Each item maps to a label in the diagrams above.

1. **Token embedding** `W_E ∈ R^(V × D)`. A lookup table: token id → a `D`-vector. This is what seeds the residual stream `x₀`. Initialized small (`σ ≈ 0.02`) so it doesn't flood the stream at step 0 — [2.3/02](part2_neural_network_fundamentals/2.3_init_normalization/02_xavier_kaiming_modern.md).

2. **Residual stream.** The `(B, S, D)` tensor threading through every block, modified only by `+=`. It's the model's working memory and the central abstraction of modern Transformers — see [residual stream](part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md) and Part 3.1. Each block *reads* it (through a norm), *computes* something, and *writes back* by addition. It is never normalized in place (that's pre-norm; contrast post-norm in [2.3/04](part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md)).

3. **RMSNorm** (pre-norm placement, two per block). Normalizes the *input* to each sublayer to a fixed scale so the sublayer's weights always see well-conditioned input regardless of how large the stream has grown with depth. RMSNorm drops LayerNorm's mean-centering — just `x / RMS(x) · γ` — which is cheaper and works as well at scale ([2.3/03](part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md), Part 3.2). Learnable: a per-channel gain `γ ∈ R^D`. No bias.

4. **Q/K/V projections** `W_Q, W_K, W_V`. Linear maps from the normalized stream to queries, keys, values. With **GQA** (Part 7.1), there are `H = 32` query heads but only `n_kv = 8` key/value groups — each K/V is shared across `H/n_kv = 4` query heads. This shrinks the **KV cache** at inference (the thing that dominates decode memory, Part 9.2) for almost no quality loss. Hence `W_K, W_V` are 4× smaller than `W_Q` in the param table below.

5. **RoPE** (rotary position embedding), applied to `q` and `k` only, right before the dot product. Position enters as a *rotation* of the query/key vectors by an angle proportional to position, so the score `q·k` depends only on *relative* position. This is why no positional vector is added at the input. RoPE is the modern default; its context-extension variants (YaRN, NTK) are Part 5.3 / 7.5.

6. **QK-norm** (optional, increasingly standard). RMSNorm applied to `q` and `k` per head before the dot product. The attention logit `q·k/√D_h` is *unbounded* — pre-norm caps the block input but nothing caps the post-projection magnitudes, so logits can grow during training until softmax saturates or fp16 overflows. QK-norm caps them directly. (Gemma's logit soft-capping is a `tanh` alternative.) This is the fix for the attention-logit-growth instability; see Part 5.1 / 7.

7. **Scaled dot-product attention** `A = softmax(QKᵀ/√D_h + mask) `, then `A V`. The `/√D_h` calibrates logit variance at init (Part 5.1). The **causal mask** sets future positions to `-∞` so position `t` attends only to `≤ t`. The `(B, H, S, S)` score tensor is the quadratic-in-`S` memory hog that FlashAttention (Part 7.2) avoids materializing — and the bulk of what gradient checkpointing frees ([2.2/04](part2_neural_network_fundamentals/2.2_backpropagation/04_gradient_checkpointing.md)).

8. **Output projection** `W_O`. Mixes the concatenated per-head outputs back to `D` and writes into the residual stream. Often initialized with extra `1/√(2L)` damping so the `L` residual additions don't compound at init ([2.3/02](part2_neural_network_fundamentals/2.3_init_normalization/02_xavier_kaiming_modern.md)).

9. **SwiGLU FFN.** The per-token MLP, where most of the parameters and FLOPs live. Three matrices (`W_gate`, `W_up`, `W_down`) instead of the classic two — see the activation aside. Acts independently on each position (no token mixing; that's attention's job).

10. **Final RMSNorm + Unembedding.** One last norm, then `W_U ∈ R^(D × V)` maps each position's `D`-vector to `V` logits. `W_U` is frequently **tied** to `W_Eᵀ` (same matrix, transposed) — saves `V·D` parameters and ties input/output token representations. Softmax over the logits gives the next-token distribution; cross-entropy against the true next token is the loss (Part 1.2, Part 6.1).

---

## Aside — the activation function: why SwiGLU, why three matrices

The classic Transformer FFN is `W_down · GELU(W_up · x)` — two matrices, one nonlinearity, hidden width `4D`. Modern models replace it with **SwiGLU**, a *gated* variant:

```
SwiGLU(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down        SiLU(z) = z · σ(z)
```

- **Gating.** One linear branch (`gate`) is squashed by SiLU and used to *modulate* the other linear branch (`up`) elementwise. The network learns, per channel, how much of `up` to let through — a multiplicative interaction the plain `GELU(·)` MLP can't express. Empirically a consistent quality win ([2.1/02](part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md), [SwiGLU deep-dive](part1_math_foundations/1.1_linear_algebra/supplementary/02_swiglu.md), Part 5.2).
- **Why SiLU and not ReLU here?** SiLU (`x·σ(x)`, a.k.a. swish) is smooth and, unlike ReLU, has no hard zero region — its gradient is small but nonzero for negative inputs, so it avoids the dead-unit failure mode. As the *gate* it gives a soft, differentiable 0→1 valve.
- **The three-matrix cost and the `2/3` rule.** Gating adds a third weight matrix (`W_gate`), so to keep the FFN's parameter count matched to a classic `4D` MLP, the hidden width is set to `≈ 2/3 · 4D = 8D/3` instead of `4D`. For `D = 4096` that's `≈ 10923`, rounded to a hardware-friendly `14336` here. This is the "`~2.7D`" in the [NOTATION.md](NOTATION.md) `d_ff` entry.

---

## Aside — what's actually learnable

Not everything in the diagram is a trained parameter. Worth being explicit, because it clarifies what gradients flow into:

| Learnable (has gradients) | **Not** learnable (fixed function / hyperparameter) |
|---|---|
| `W_E` (embedding), `W_U` (unembed, if untied) | RoPE rotation angles (deterministic function of position) |
| `W_Q, W_K, W_V, W_O` per block | Causal mask |
| `W_gate, W_up, W_down` per block | The `√D_h` attention scale |
| RMSNorm gains `γ` (one per norm) | Softmax, SiLU, the `⊙` and `+` ops |
| Final RMSNorm gain `γ_f` | Layer count `L`, widths `D`, `d_ff`, head counts |

Note what's **absent** versus the BERT-era model you remember: **no bias terms** anywhere (dropped as needless params that can destabilize, [2.1/03](part2_neural_network_fundamentals/2.1_mlp_building_block/03_bias_terms.md)), **no learned positional embedding table** (RoPE replaced it), and **no LayerNorm mean/bias** (RMSNorm has gain only).

---

## Aside — where the parameters live (8B-class worked example)

Config: `D=4096, L=32, H=32, D_h=128, n_kv=8, d_ff=14336, V=128000`.

**Per block:**

| Tensor | Shape (row-conv) | Params |
|---|---|---|
| `W_Q` | `D × (H·D_h)` = 4096 × 4096 | 16.8M |
| `W_K` | `D × (n_kv·D_h)` = 4096 × 1024 | 4.2M |
| `W_V` | `D × (n_kv·D_h)` = 4096 × 1024 | 4.2M |
| `W_O` | `(H·D_h) × D` = 4096 × 4096 | 16.8M |
| **Attention subtotal** | | **42.0M** |
| `W_gate` | `D × d_ff` = 4096 × 14336 | 58.7M |
| `W_up` | `D × d_ff` = 4096 × 14336 | 58.7M |
| `W_down` | `d_ff × D` = 14336 × 4096 | 58.7M |
| **FFN subtotal** | | **176.2M** |
| 2 × RMSNorm `γ` | 2 × 4096 | 0.008M |
| **Block total** | | **≈ 218M** |

**Whole model:**

| Part | Params |
|---|---|
| Token embedding `W_E` (V·D) | 524M |
| Blocks (218M × 32) | 6.98B |
| Final RMSNorm | ~0 |
| Unembedding `W_U` (if **untied**) | 524M |
| **Total, tied embeddings** | **≈ 7.5B** |
| **Total, untied embeddings** | **≈ 8.0B** |

Three things to take away from the table:

- **The FFN dominates each block** — 176M of 218M, ~80%. Attention is only ~20% of the *parameters* (though its `(B,H,S,S)` activations and KV cache dominate *memory*). When people say "most of an LLM's weights are in the MLPs," this is the arithmetic.
- **GQA's savings are visible**: `W_K/W_V` are 4.2M each instead of 16.8M (the `n_kv=8` vs `H=32` ratio). Without GQA, attention would be ~67M/block.
- **Norms are a rounding error** on parameter count (8K/block) — their value is stability, not capacity.

For how this parameter count translates into *training memory* (params + grads + Adam states + activations, and why activations dominate), see the gradient-checkpointing walkthrough ([2.2/04](part2_neural_network_fundamentals/2.2_backpropagation/04_gradient_checkpointing.md)).

---

## What changed from the BERT-era Transformer you remember

| 2018-era (BERT / original Transformer) | Modern decoder-only LLM |
|---|---|
| Post-norm (`LN(x + sublayer(x))`) | **Pre-norm** (`x + sublayer(LN(x))`) — trains deep without warmup babysitting |
| LayerNorm (mean-center + scale + bias) | **RMSNorm** (scale only) — cheaper, no mean-centering |
| Learned/sinusoidal absolute positions added at input | **RoPE** rotation inside attention — relative, extrapolates better |
| Bias terms everywhere | **No biases** |
| `ReLU`/`GELU` two-matrix FFN, width `4D` | **SwiGLU** three-matrix gated FFN, width `~2.7D` |
| Full multi-head attention | **GQA** (shared K/V) — shrinks the KV cache for inference |
| Encoder (bidirectional) or enc-dec | **Decoder-only**, causal mask, next-token objective |
| Dropout on by default | Dropout largely **gone** ([2.5/01](part2_neural_network_fundamentals/2.5_regularization/01_dropout.md)) |

The skeleton — attention + FFN + residual + norm, repeated `L` times — is unchanged. Everything that moved is a stability, efficiency, or inference-cost refinement on that skeleton.

---

## Self-check

1. Position information is not added to the residual stream at the input. Where does it enter, and why is "relative position" a consequence of that mechanism?
2. The parameter table shows attention at ~20% of each block's weights but it dominates *memory* during training and inference. Name the two different tensors responsible (one for each).
3. Pre-norm never normalizes the residual stream itself. So what stops the stream's magnitude — which grows with depth — from making the final logits explode at initialization?
4. SwiGLU uses three weight matrices but is tuned to match a two-matrix `4D` FFN's parameter count. What width does that force, and why that specific ratio?

### Answers

1. RoPE rotates `q` and `k` inside attention by an angle proportional to their absolute positions, so the score `q·k` depends only on the *difference* of the two angles — i.e. the relative offset between query and key. Position never exists as an additive vector in the stream; it only shapes attention scores.
2. **Training memory**: the `(B, H, S, S)` attention score/probability tensor — quadratic in sequence length, cached for backward (what FlashAttention and gradient checkpointing target). **Inference memory**: the **KV cache** — the stored `K, V` for all past tokens, which GQA shrinks by sharing K/V across query heads.
3. Initialization. The output projections (`W_O`, `W_down`) that write into the stream are scaled down — the `0.02` constant, and explicitly `1/√(2L)` damping — so each block's additive contribution is small and the `L`-term sum stays O(1) at step 0. RMSNorm then protects each *sublayer input* regardless of stream scale, but it's the init that keeps the raw stream (and thus the final logits) bounded. Erring small is safe (the next norm renormalizes); erring large compounds across `L` additions.
4. Width `≈ 2/3 · 4D = 8D/3` (≈ `2.7D`). Gating adds a third `D × d_ff` matrix, so three matrices at width `8D/3` cost about the same as two matrices at width `4D` (`3 · 8/3 = 8 = 2 · 4`). The `2/3` exactly compensates for the extra matrix, keeping FLOPs and parameters comparable to the classic FFN.
