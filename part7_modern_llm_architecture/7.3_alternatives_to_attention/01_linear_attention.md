# Linear Attention — Where the Recurrence Comes Back

Part 4 closed with a promise: the RNN's compact evolving state wasn't wrong, only mismatched to 2017 hardware, and something would try to win back the `O(1)`-memory inference the Transformer gave up ([4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)). This is that file. The striking part is how *little* you have to change: delete one nonlinearity from attention and a recurrent network with a fixed-size state falls out automatically.

## Delete the softmax, reassociate the matmul

Standard attention, ignoring scaling, computes `softmax(QKᵀ) V`. The `softmax` is what forces you to form `QKᵀ` — an `S × S` object — before you can do anything else. Replace `exp(qᵢ · kⱼ)` with a factored similarity `φ(qᵢ) · φ(kⱼ)` for some feature map `φ` (Katharopoulos et al., 2020, using `φ(x) = elu(x) + 1` to keep things positive), and the expression becomes a plain product of three matrices:

```
softmax attention:   softmax(Q Kᵀ) V              must form (S × S)
linear attention:    (φ(Q) φ(K)ᵀ) V   =   φ(Q) (φ(K)ᵀ V)
                     └── O(S²·D) ──┘       └── O(S·D²) ──┘
```

Matrix multiplication is associative, so you may bracket it the other way — and the right bracketing never forms an `S × S` matrix at all. `φ(K)ᵀ V` is `D × D`, independent of sequence length. Cost goes from quadratic to **linear in `S`**, and that is the entire trick.

## The recurrent form, and why it's the same thing

For causal (autoregressive) models you can't just multiply the whole thing — position `t` may only see `≤ t`. Writing the causal version out reveals a recurrence:

```
state:   Sₜ = Sₜ₋₁ + φ(kₜ) vₜᵀ          # a D × D matrix, accumulated
norm:    zₜ = zₜ₋₁ + φ(kₜ)              # a D vector
output:  oₜ = (φ(qₜ) Sₜ) / (φ(qₜ) · zₜ)
```

That is an RNN. A fixed-size state, updated additively, read by a query. **Verified identical to the quadratic form** — max absolute difference `3.9e-16` over `S = 200`, `D_h = 32` in float64. Same math, two implementations:

| Form | Train cost | Inference per token | State |
|---|---|---|---|
| Quadratic `(φQ φKᵀ)V` | `O(S²D)`, parallel | — | — |
| Recurrent | `O(SD²)`, sequential | **`O(D²)` compute, `O(D²)` memory** | fixed `D × D` |

The inference column is the prize: **constant memory and constant per-token compute, no growing KV cache.** For `D_h = 32` that state is 1,024 numbers, forever, whether you're at token 10 or token 10 million. Compare against [7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)'s 40 GiB.

And note this resolves Part 4's tension: you can **train with the parallel form and serve with the recurrent one**, because they compute the same function. The RNN's fatal flaw was sequential training ([4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)); linear attention escapes it by having a parallel twin. That duality is the whole reason this family became viable.

## Why it lost the first time

Linear attention has been available since 2020 and did not take over. Three reasons, and the first is fundamental:

**1. The state is a fixed-size bottleneck.** `Sₜ` is `D × D` numbers regardless of sequence length, so it cannot store `S` distinct key-value associations once `S` exceeds its capacity. Softmax attention *keeps every token* and can retrieve any of them exactly; linear attention **superposes** all associations into one matrix — literally `Σₜ φ(kₜ) vₜᵀ`, a sum of outer products, which is the low-rank superposition situation from [1.4/01](../../part1_math_foundations/1.4_optional_deeper_knowledge/01_dimension_span_and_rank.md). Retrieval works while the keys stay near-orthogonal and degrades as the state saturates. This is the pigeonhole principle from [4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md), reappearing in new clothing — and it is why in-context recall is the family's characteristic weakness ([file 03](03_hybrids_and_recall.md)).

**2. It never forgets.** The plain update `Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ` adds every token with equal weight forever. Softmax's exponential can *concentrate* on a few tokens and effectively ignore the rest; a uniform sum cannot. This is exactly the LSTM's motivation for a forget gate ([4.1/02](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/02_vanishing_gradient_and_gating.md)) — and unsurprisingly, every modern variant adds decay or gating back:
   - **Gated/decaying variants** — RetNet's fixed exponential decay, GLA (gated linear attention) with learned data-dependent gates, DeltaNet's delta-rule update that *replaces* rather than accumulates.
   - Adding a gate is the single most important upgrade to the 2020 formulation, and it's what makes [file 02](02_state_space_models.md)'s Mamba work.

**3. FlashAttention moved the goalposts.** Linear attention's selling point was asymptotic, but at the sequence lengths anyone trained on (2–8K in 2022), exact attention with a good kernel was simply faster in wall-clock and better in quality ([7.2/02](../7.2_efficient_attention/02_flashattention.md)). The `O(S²)` constant was small and the `O(SD²)` constant was not.

## Why it's back

Three things changed. Context lengths grew until the asymptotics genuinely bite; **gating** fixed the never-forget problem; and — the conceptual unification — **Mamba-2 showed that state-space models and linear attention are the same family** (its "structured state-space duality"), which merged two research lines and gave the whole area hardware-efficient training algorithms. That's [file 02](02_state_space_models.md).

There are now production-scale deployments (MiniMax-01's lightning attention at ~456B total parameters), and the honest status is: **viable, mostly in hybrids, not dominant** ([file 03](03_hybrids_and_recall.md)).

## Why it matters in modern LLM work

- **It's the cleanest statement of the core tradeoff in sequence modeling**: keep everything and pay `O(S)` memory, or compress into a fixed state and accept a recall ceiling. Every architecture in this subsection is a point on that line.
- **The parallel/recurrent duality is the key enabling idea** and recurs throughout 7.3 — any model with both forms gets fast training *and* cheap inference.
- **"Superposition into a fixed state" connects to the residual-stream story** ([1.4/04](../../part1_math_foundations/1.4_optional_deeper_knowledge/04_linear_readout_and_identifiability.md)): the recall limit is a linear-readout limit, and the same near-orthogonality arithmetic governs both.

## Self-check

1. What single change to attention makes reassociation legal, and why does the bracketing matter so much?
2. Write the recurrent update and identify which part is the "KV cache equivalent." How big is it?
3. Linear attention has both a parallel and a recurrent form. Why is having both essential rather than merely convenient?
4. State the recall limitation mechanically — what object saturates, and what is it a sum of?
5. Why does every modern linear-attention variant add gating, and which Part 4 mechanism is that?
6. Give the two reasons linear attention lost in 2020–22 that were *not* about its math.

### Answers

1. Removing the **softmax** — replacing `exp(qᵢ·kⱼ)` with a factored `φ(qᵢ)·φ(kⱼ)`. Softmax is a nonlinear function *of the full row*, so `QKᵀ` must exist before it can be applied; a factored similarity leaves a plain triple matrix product, which associativity lets you bracket as `φ(Q)(φ(K)ᵀV)`. That bracketing never forms an `S × S` matrix — the `D × D` intermediate is independent of sequence length, so cost drops from `O(S²D)` to `O(SD²)`.
2. `Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ` (plus a normalizer `zₜ`), read as `oₜ = φ(qₜ)Sₜ / (φ(qₜ)·zₜ)`. The state `Sₜ` is the KV-cache equivalent: a fixed `D × D` matrix — 1,024 values at `D_h = 32` — constant in `S`, versus a cache growing 320 KiB per token ([7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)).
3. Because the two phases have opposite needs. Training must parallelize across the sequence or it's economically dead — the RNN's fatal flaw ([4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)) — while inference wants `O(1)` state and per-token cost. Having mathematically equivalent forms lets you use the parallel one to train and the recurrent one to serve, capturing both columns of Part 4's tradeoff table instead of choosing.
4. The state `Sₜ = Σₜ φ(kₜ)vₜᵀ` is a **sum of outer products** — a single `D × D` matrix superposing every key-value association ever seen. Its rank is capped at `D`, so once the number of distinct associations exceeds what nearly-orthogonal keys can keep separable, retrievals interfere. Exact recall of an arbitrary past token is impossible in principle, not just hard.
5. Because the plain update weights every past token equally and forever, so the state saturates and cannot concentrate on what matters — while softmax's exponential *can* sharply prefer a few tokens. Gating restores the ability to decay or overwrite: it is the **LSTM forget gate** ([4.1/02](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/02_vanishing_gradient_and_gating.md)), rediscovered, and it's the difference between the 2020 formulation and the ones that work.
6. (a) **FlashAttention** made exact attention fast and linear-memory, so the asymptotic argument stopped paying at the `S` values in use — constants dominated ([7.2/02](../7.2_efficient_attention/02_flashattention.md)). (b) **Context lengths were short** (2–8K), so `O(S²)` simply wasn't the bottleneck yet. Both are about the empirical regime, not about linear attention being wrong.

## Exercise

Verify the equivalence and then find the ceiling. (a) Implement both forms — masked quadratic `((φQ φKᵀ) ⊙ causal)V` normalized, and the `O(S)` recurrence — and confirm they match to ~1e-16 in float64 at `S = 200`, `D_h = 32`. (b) Now measure the recall limit directly: build an associative-recall task with `n` random key-value pairs written into the sequence followed by a query for one of them, and plot retrieval accuracy against `n` for `D_h ∈ {16, 32, 64}`. You should find accuracy collapses as `n` approaches roughly `D_h`, and that softmax attention on the same task stays flat — the pigeonhole ceiling, measured. (c) Add exponential decay (`Sₜ = γSₜ₋₁ + φ(kₜ)vₜᵀ`) and describe how the failure mode changes: what does the model now get right, and what does it now get wrong?
