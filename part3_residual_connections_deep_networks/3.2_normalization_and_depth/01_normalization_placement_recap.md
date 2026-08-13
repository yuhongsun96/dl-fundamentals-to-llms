# Normalization Placement, Revisited

You've already met pre- vs. post-norm as an *initialization/optimization* story in [2.3/04](../../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md) and the norm *mechanics* (LayerNorm/RMSNorm) in [2.3/03](../../part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md). This file doesn't re-derive either — read those first. Its job is to re-tell placement as a **residual-stream** story (now that [3.1/04](../3.1_skip_connection/04_residual_stream_as_abstraction.md) gives us the stream) and to collect the **at-scale stabilizers** that live at the residual/norm interface. If pre-norm already feels settled, skim to the stabilizer table.

**Convention:** per-token stream `h_l ∈ R^D`, sublayer `f_l` (attention or FFN). "Norm" = LayerNorm or RMSNorm; placement is what's at issue, not which one.

## The two placements, in residual-stream terms

```
post-norm (2017):   h_{l+1} = Norm( h_l + f_l(h_l) )        # normalize the STREAM itself
pre-norm  (modern): h_{l+1} = h_l + f_l( Norm(h_l) )        # normalize only the COPY f reads
```

The [3.1/04](../3.1_skip_connection/04_residual_stream_as_abstraction.md) reframing makes the choice obvious in a way the optimization argument didn't:

- **The residual stream is the shared bus every component reads and writes** (`h_l = embed + Σ contributions`). Its additive-accumulation semantics are the whole abstraction.
- **Post-norm normalizes that accumulated sum at every layer** — it rescales and re-centers the bus in place, partially *erasing* the magnitude information earlier layers wrote. The clean identity highway of [3.1/02](../3.1_skip_connection/02_gradient_highway.md) is routed *through* a norm at each step: the backward `I·I·⋯·I` term becomes `J_norm·J_norm·⋯` — a product of `L` norm-Jacobians, which is exactly the multiplicative compounding residuals were supposed to remove.
- **Pre-norm leaves the stream untouched** and normalizes only the *copy* a sublayer reads. The bus keeps its clean additive semantics; the highway stays a pure identity path; each sublayer still sees a well-conditioned unit-scale input regardless of how big the stream has grown.

So the residual-stream view and the gradient view agree: **don't normalize the highway.** Pre-norm is "normalize the reads, not the bus." That's the one-sentence reason GPT-2 onward, LLaMA, Mistral, Qwen, and essentially every modern decoder use it.

## The cost pre-norm pays (and its two fixes)

Pre-norm's price, from [2.3/04](../../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md): because the stream is never normalized in place, its **magnitude grows with depth** (`Var ~ L`). That single fact drives two design elements you'll always see together with pre-norm:

1. **A final norm before the unembedding.** The stream reaching `W_U` has `L` accumulated contributions; one last norm tames its scale so logits are well-conditioned. (Post-norm doesn't need this — its blocks already end in a norm.)
2. **Output-projection down-scaling at init** (`1/√(2L)` on `W_O`, `W_down`), so the `L` additions don't compound at step 0.

The variance-growth problem and its full set of remedies (init scaling, LayerScale, ReZero) are the entire subject of [3.2/02](02_scaling_the_residual_stream.md) — this file just flags that the *final norm* is the placement-level piece of that story.

## At-scale stabilizers (the residual/norm interface)

Pre-norm makes deep stacks train, but at frontier scale (100B+ params, long runs) new instabilities appear — loss spikes, attention-logit blowups ([2.2/05](../../part2_neural_network_fundamentals/2.2_backpropagation/05_gradient_pathologies.md)). The fixes cluster at the norm/residual interface:

| Technique | What it does | Where it acts | Status |
|---|---|---|---|
| **Pre-norm + final norm** | Normalize reads, tame accumulated stream before logits | Every sublayer input + once at the end | **Default** (LLaMA, GPT, Mistral, Qwen) |
| **DeepNorm** (2022) | Post-norm made stable at depth by up-scaling the residual: `Norm(α·h + f(h))`, `α=(2L)^{1/4}` | The residual add | Niche (1000-layer experiments); pre-norm usually preferred — [2.3/04](../../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md) |
| **Sandwich-LN** (2022) | Norm *both* before and after the sublayer: `h + Norm(f(Norm(h)))` — damps each write's magnitude before it hits the stream | Around the sublayer | Some Google models, Gemma-2 — [2.3/04](../../part2_neural_network_fundamentals/2.3_init_normalization/04_pre_vs_post_norm.md) |
| **QK-norm** (2023) | RMSNorm on `q` and `k` per head before the attention dot product | Inside attention, pre-`QKᵀ` | Increasingly standard (Gemma-2, some Qwen, Chameleon) — see below |
| **Logit soft-capping** | `tanh` cap on attention and/or final logits: `c·tanh(x/c)` | Attention scores, output logits | Gemma-2; a `tanh` alternative to QK-norm |

The last two target a specific instability pre-norm does **not** cover, worth spelling out because it's a common "wait, why is it still blowing up?" moment:

> **Pre-norm caps a sublayer's *input* magnitude, but nothing caps the *output* of the Q/K projections.** The attention logit `q·kᵀ/√D_h` is unbounded above — as `W_Q, W_K` grow during training, logits can grow until softmax saturates (gradients vanish) or fp16 overflows (NaN). This "attention-logit growth" is a real frontier-scale failure mode.

- **QK-norm** fixes it directly: normalize `q` and `k` to unit scale per head *after* their projections and *before* the dot product, so the logit magnitude can't run away regardless of how large `W_Q, W_K` get. It sits *inside* attention, downstream of the pre-norm that only protected the block input. (Also noted in [ARCHITECTURE.md](../../ARCHITECTURE.md) component 6 and [2.3/03](../../part2_neural_network_fundamentals/2.3_init_normalization/03_norm_layers.md).)
- **Logit soft-capping** is the alternative: instead of normalizing the inputs to the dot product, squash the output with `c·tanh(score/c)`, which is smooth and bounds scores to `±c`. Gemma-2 used it (and later moved partly back toward QK-norm-style approaches). QK-norm is now the more common choice because `tanh` capping interacts awkwardly with FlashAttention (Part 7.2).

The meta-point: **norm placement isn't one decision, it's a stack of them** — where to normalize the block input (pre vs. post), whether to also normalize the block output (sandwich), whether to normalize inside attention (QK-norm), and whether to cap logits. Modern models mix and match. But the load-bearing choice — the one that makes deep training possible at all — is *pre-norm on the block input*, and the reason is: keep the residual highway clean.

## Why this matters for modern NLP/LLM work

- Reading any post-2020 architecture, you can immediately place its norm choices on this table and know *what instability each one targets* — pre-norm (depth), final norm (accumulated stream), QK-norm/soft-cap (attention-logit growth), sandwich (per-write magnitude).
- "Pre-norm keeps the residual highway clean" is the unifying sentence that connects [3.1](../3.1_skip_connection/02_gradient_highway.md), norm placement, and the scaling remedies in [3.2/02](02_scaling_the_residual_stream.md).
- When a large run destabilizes, this table is the menu of fixes — and knowing that pre-norm alone doesn't stop attention-logit blowup saves you from misdiagnosing it.

## Self-check

1. Restate why pre-norm beats post-norm using the residual-*stream* abstraction (not the gradient argument from Part 2). What does post-norm do to the shared bus?
2. Pre-norm needs a *final* norm before the unembedding; post-norm doesn't. Explain both, in terms of what the stream looks like at the output.
3. A model uses pre-norm and *still* NaNs from attention-logit overflow deep in training. Why didn't pre-norm prevent this, and name the two fixes that would.
4. QK-norm and logit soft-capping target the same instability. How does each work, and why is QK-norm now often preferred?

### Answers

1. The residual stream is a shared additive bus whose semantics depend on contributions *accumulating* undisturbed. Post-norm re-centers and re-scales that bus *in place* at every layer, partially erasing the magnitude information earlier components wrote and (equivalently) routing the identity highway through a norm at each step. Pre-norm normalizes only the *copy* each sublayer reads, leaving the bus untouched — so accumulation semantics and the clean identity path both survive. "Don't normalize the highway."
2. In pre-norm the output stream is the raw sum of `L` block contributions plus the embedding — un-normalized, with depth-dependent magnitude — so it needs one final norm to condition it before `W_U` produces logits. In post-norm every block *ends* in a norm, so the output stream is already normalized; a separate final norm would be redundant.
3. Pre-norm caps each sublayer's *input* magnitude, but the Q/K *projections* produce unbounded outputs, so `q·kᵀ/√D_h` can grow without limit as `W_Q, W_K` train — softmax saturates or fp16 overflows. Fixes: **QK-norm** (normalize `q,k` per head before the dot product) or **logit soft-capping** (`c·tanh(score/c)`).
4. QK-norm normalizes the *inputs* to the attention dot product (unit-scale `q,k` per head), bounding the logit indirectly; soft-capping squashes the *output* score through `c·tanh(·/c)`, bounding it directly to `±c`. QK-norm is often preferred because `tanh` capping composes poorly with FlashAttention's fused kernels, whereas normalizing `q,k` beforehand does not.

## Exercise

Take the 3-block chain from [3.1/02](../3.1_skip_connection/02_gradient_highway.md)'s exercise. (a) Write the backward `∂h_3/∂h_0` for the *post-norm* version `h_{l+1} = Norm(h_l + f_l(h_l))` and identify what has replaced each identity `I` from the pre-norm expansion. (b) Explain, in one sentence, why the product of those replacement factors across `L=100` layers is the trainability problem. (c) For a pre-norm model, add QK-norm to one attention sublayer and state precisely which quantity's magnitude is now bounded that wasn't before.
