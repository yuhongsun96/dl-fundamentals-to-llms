# The Feed-Forward Network

The block's second sublayer, and — surprisingly — where most of the model lives. It is the least glamorous component (an ordinary MLP, no attention magic) and yet it holds ~80% of a block's parameters and does the bulk of its FLOPs. This file is about *what the FFN is for*, why it's shaped the way it is, and the memory-retrieval interpretation that reframes it as attention's complement rather than an afterthought.

**Convention:** row-vector (`Y = X W`), repo default — `(B, S, D) @ (D, d_out) → (B, S, d_out)`. Anchored to the [ARCHITECTURE.md](../../ARCHITECTURE.md) 8B config: `D = 4096`, `d_ff = 14336`, `L = 32`.

## What it is

The classic Transformer FFN is a two-matrix MLP with one nonlinearity, applied at every position:

```
FFN(x) = φ(x W_up) W_down          x: (B,S,D)
         └─ D → d_ff ─┘ └ d_ff → D ┘
```

with `W_up ∈ R^(D, d_ff)`, `W_down ∈ R^(d_ff, D)`, and `φ` a pointwise nonlinearity (GELU in the BERT/GPT era; see [2.1/02](../../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md)). Widen from `D` to a larger `d_ff`, bend, shrink back to `D`. The output re-enters the residual stream by addition, same as attention.

Two properties define its role in the block, and both are the point of the previous file:

- **Per-position.** The *same* `W_up, W_down` are applied independently to each of the `S` tokens. Position `t` in, position `t` out — the FFN never looks at another position. That's deliberate: cross-token movement is attention's job ([5.1/03](../5.1_self_attention/03_attention_as_lookup.md) and [5.2/01](01_assembling_the_block.md)). The FFN is the block's per-token *computation*, attention its cross-token *communication*.
- **Channel-mixing and nonlinear.** Within a token's `D`-vector, the FFN mixes channels and applies a nonlinearity — the only genuinely per-position nonlinear transform in the block.

## Why the width expands: capacity

The classic choice is `d_ff = 4D` — the hidden layer is four times wider than the model dimension. Why widen at all, and why by 4×?

The widening is where the FFN's **capacity** lives. Projecting into a `4D`-dimensional space before the nonlinearity gives the layer room to represent many more distinct features than `D` channels could, apply the nonlinearity in that expanded space, then compress the result back. A wider hidden layer = more feature detectors = more the layer can compute. The `4×` is an empirical sweet spot inherited from the original Transformer that has held up across a decade of scaling (with the SwiGLU adjustment below).

The consequence is that **the FFN dominates the block's parameter and FLOP budget.** From the [ARCHITECTURE.md](../../ARCHITECTURE.md) 8B param table:

| Sublayer | Params per block | Share |
|---|---|---|
| Attention (`W_Q, W_K, W_V, W_O`, with GQA) | 42.0M | ~19% |
| **FFN** (`W_gate, W_up, W_down`) | **176.2M** | **~81%** |
| 2× RMSNorm | 0.008M | ~0% |
| Block total | ≈ 218M | |

So when people say "most of an LLM's weights are in the MLPs," this table is the arithmetic: ~176M of a block's ~218M parameters, ~80%, are the FFN. Attention gets all the attention, but the FFN is where the parameters — and, at typical sequence lengths, most of the compute — actually go. (Attention's cost dominates *memory* through the `(B,H,S,S)` scores and the KV cache, but not *parameters*.)

## The key-value memory interpretation

A more useful mental model than "an MLP for capacity" comes from Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories*. Split the FFN at its nonlinearity and read the two matrices by their rows and columns:

```
      x W_up                φ(·)              (·) W_down
D-vec ──────► d_ff scores ─────► d_ff gates ──────────► D-vec
      keys                                    values
```

- **`W_up` rows are keys.** Each of the `d_ff` rows of `W_up` is a `D`-dimensional pattern. The product `x W_up` scores the input token against all `d_ff` keys — coordinate `i` of the hidden vector is `⟨x, W_up[i,:]⟩`, how strongly key `i` fires for this token. After the nonlinearity, only some keys are "on." Empirically these keys are interpretable: individual hidden units detect concrete patterns — "sentence ends in a date," "token is inside a URL," "previous words describe a color."
- **`W_down` columns are values.** Each of the `d_ff` columns of `W_down` is a `D`-dimensional vector *written back into the residual stream* when its key fires. So the FFN output is a gated sum of values: `Σᵢ gateᵢ · W_down[i,:]` — the retrieved memories for whichever keys the token activated.

Read this way, the FFN is an **associative memory**: it detects patterns in the incoming token (keys) and writes learned responses back to the stream (values). This is the complement to attention's dynamic routing. Attention's "keys/values" are *other tokens* and its weights are computed fresh per input; the FFN's keys/values are *fixed learned parameters* and its lookup is against a static memory. Attention retrieves from the sequence; the FFN retrieves from the weights. Both write to the same residual stream. (That the FFN holds ~80% of the parameters now reads as: most of what the model *knows* — factual and pattern associations — is stored in these key-value memories, which is consistent with knowledge-editing work like ROME/MEMIT that surgically edits FFN weights to change a stored fact.)

## Gated variants: what modern LLMs actually use

Every frontier open LLM has replaced the two-matrix FFN with a **gated** three-matrix variant, SwiGLU:

```
SwiGLU FFN(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down       SiLU(z) = z·σ(z)
```

Project `x` two ways in parallel; squash one branch (`gate`) through SiLU; multiply the two branches elementwise; project the product back down. The gate learns, per channel per position, how much of the `up` branch to let through — a multiplicative interaction the plain `φ(x W_up)` MLP cannot express.

The full story — why gating helps (multiplicative interactions are expressive in a way summation isn't), why SiLU rather than ReLU as the gate, and the exact bookkeeping behind the width adjustment — is worked in detail in [2.1/02](../../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md). Do not re-derive it here; the two takeaways to carry are:

- **Gating buys expressivity per parameter, not more parameters.** SwiGLU beats GELU FFNs at *matched* parameter and FLOP budget. The win is architectural, not just "bigger."
- **The `2/3` width keeps it FLOP-matched.** The third matrix (`W_gate`) would make three `4D`-width matrices cost 1.5× a two-matrix `4D` FFN. So `d_ff` is shrunk to `≈ 2/3 · 4D = 8D/3 ≈ 2.7D`, and three skinnier matrices cost what two fatter ones did. For `D = 4096`, that's `≈ 10923`, rounded up to the hardware-friendly `14336` in the 8B config. The key-value-memory reading still applies — there are just now two key matrices (`W_gate` as the gating key, `W_up` as the value-selecting key) whose product forms the retrieval.

Note the param table above already uses the SwiGLU shape: three matrices of 58.7M each (`4096 × 14336` and its transpose) = 176.2M.

### At a glance: what each era / model uses

| Model | FFN form | Nonlinearity | Width |
|---|---|---|---|
| Original Transformer (2017) | 2-matrix | ReLU | `4D` |
| BERT, GPT-2, GPT-3, T5, OPT | 2-matrix | GELU | `4D` |
| PaLM | gated (GeGLU) | GELU-gated | `~2.7D` |
| Llama 1/2/3, Mistral, Mixtral | gated (SwiGLU) | SiLU-gated | `~2.7D` |
| Qwen 1/2/2.5, DeepSeek-V2/V3 | gated (SwiGLU) | SiLU-gated | `~2.7D` |
| Gemma 1/2 | gated (GeGLU) | GELU-gated | `~2.7D` |
| Kimi K3 (2026) | gated (SiTU-GLU) | soft-capped SwiGLU | MoE experts |

The split is clean: **GELU + `4D` two-matrix** is the 2018–2021 era; **gated (SwiGLU/GeGLU) + `~2.7D` three-matrix** is everything from Llama-1 onward. If a 2024 model card doesn't say, assume SwiGLU. (Mixture-of-Experts models replace this single FFN with many FFN "experts" plus a router — same per-position, key-value-memory FFN, just `N` of them with sparse selection; that's Part 7.)

## Self-check

1. The FFN is applied "position-wise." What breaks if you instead let it mix across positions — and what component already does that job?
2. In the key-value view, which matrix holds the keys and which the values, and what does the nonlinearity do between them?
3. The 8B FFN uses three matrices of 58.7M params each, yet is described as "matched" to a classic two-matrix `4D` FFN. Show the arithmetic that makes three matrices cost the same as two.

### Answers

1. Nothing *breaks* mathematically, but you'd be duplicating attention's role and destroying the block's clean division of labor — cross-position mixing is exactly what self-attention exists to do ([5.2/01](01_assembling_the_block.md)). The FFN is kept strictly per-position so that the *only* cross-token operator in the block is attention; this keeps the two sublayers complementary (communication vs. computation) and, practically, keeps the FFN a simple batched matmul over `B·S` independent tokens.
2. **`W_up` (and, in SwiGLU, `W_gate`) rows are the keys** — `x W_up` scores the token against each of the `d_ff` key patterns. **`W_down` columns are the values** — each fired key contributes its `D`-vector back to the residual stream. The **nonlinearity** in between is the gate: it turns the raw key-match scores into activations that decide *how much* of each value gets retrieved (a hard-ish on/off for ReLU, a soft gate for GELU/SiLU). Output = gated sum of value vectors.
3. Classic FFN: two matrices at width `4D`, cost `2 · D · 4D = 8D²` (per direction). SwiGLU: three matrices, but at width `8D/3`. Cost `3 · D · (8D/3) = 8D²`. Identical. The `2/3` factor is chosen precisely so that `3 · (2/3) = 2` — the extra matrix is paid for by the narrower hidden dimension, leaving parameters and FLOPs matched to the classic block. (Full version in [2.1/02](../../part2_neural_network_fundamentals/2.1_mlp_building_block/02_activations.md), self-check #3.)

## Exercise

Instrument one FFN of a small open SwiGLU model (or a toy one you train). Feed it a batch of tokens and, for a single hidden unit `i`, find the tokens/contexts that maximize its post-activation value — i.e. the inputs for which "key `i`" fires hardest. Do the top-activating contexts share an interpretable pattern (a topic, a syntactic position, a token class)? Then look at the corresponding value vector `W_down[i,:]`: project it through the unembedding (`W_down[i,:] · W_U`) and read off which *output tokens* that value promotes when written to the stream. This is the key-value-memory view made concrete — a key that detects a pattern, wired to a value that pushes the next-token distribution in a specific direction. Contrast with attention, where the analogous "what does this retrieve" question has a different answer for every input.

> **Worked version**: [supplementary/02_ffn_key_value_memories.ipynb](supplementary/02_ffn_key_value_memories.ipynb) runs this exercise end-to-end on a real 135M SwiGLU checkpoint (SmolLM2-135M, ~270 MB, CPU, ~10 s). It finds a memory whose key is *"a monetary magnitude is in progress"* and whose value promotes `' million'`/`' billion'`/`' lakh'`, verifies the key generalizes to held-out text, then **ablates** the single unit and watches `p(' million')` fall to 0.11x. It also shows the FFN's write direction is fixed across inputs (`cos = 1.0000`) where an attention head's is not (`cos ≈ 0`) — the static-vs-dynamic retrieval contrast, measured.
