# Multi-Head Attention

This file delivers the full treatment promised by the aside in [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md) ("softmax sharpness is *not* the reason for multi-head... full treatment in Part 5.1"). Read that aside first if you haven't — it draws the crucial line between what constrains `D_h` (softmax sharpness / the `√d_head` SNR argument) and what motivates `H` (the subject of this file). We pick up exactly where it left off.

**Convention:** row-vector (`Y = X W`), repo default. Dims from [NOTATION.md](../../NOTATION.md); 8B config `D=4096, H=32, D_h=128` from [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Why H > 1: one head averages; multiple heads attend in parallel

The original motivation, from "Attention Is All You Need," is an **averaging** argument. A single attention head produces *one* softmax-weighted average of the value vectors — it attends within a single representation subspace and **blends** what it pulls in. In the paper's words, multi-head attention lets the model "jointly attend to information from **different representation subspaces at different positions**," whereas "with a single attention head, **averaging inhibits this**." The fix: run `H` heads in parallel, each with its own `(W_Q^h, W_K^h, W_V^h)` — its own subspace and its own softmax — so you get `H` separate retrievals instead of one averaged blend.

A sharper, mechanistic restatement (developed *after* the paper, not in it): a single softmax is **one normalized selection** — one ranking of the keys sharing a single mass budget that sums to 1. It *can* spread that mass over several keys, but only along **one** criterion (`q·k`), and whatever it selects is averaged through one `W_V`. What it *can't* do is run **two independent selections** — "attend to A because it's my antecedent **and**, separately, to B because it's my verb" needs two different rankings with two separate budgets. `H` heads give exactly that: `H` independent (ranking, softmax) pairs, each in its own subspace. (This is about *selection*, not storage — [01](01_qkv_projections.md): superposition already lets one vector hold many features, but one head still can't select them independently.)

Concretely, the word "it" mid-sentence may need, in parallel: its previous token, its syntactic head (the verb), its coreferent antecedent, a matching bracket, the BOS/attention sink — each a *different* key matched by a *different* criterion, fetching a *different* value. `H` heads let head 1 handle "previous token," head 5 "coreference," and so on. Note this has **nothing to do with softmax sharpness** (next section / [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)) — even a perfectly-tempered softmax is still a single selection.

**How much of this is established vs. intuition.** Treat the tidy "each head specializes in a distinct relation" story as **post-hoc intuition** — offered as motivation and only *partly* borne out. The evidence cuts both ways: interpretability finds genuinely specialized heads (previous-token, induction, syntactic, name-mover — Part 11.2), *while* head-pruning studies (Michel et al. 2019, "Are Sixteen Heads Really Better than One?"; Voita et al. 2019) show most heads are **redundant** and can be dropped after training with little loss, importance concentrated in a few. So the well-supported, load-bearing claim is the *capability* — `H` independent parallel selections at ~no extra parameter cost — not that every head learns a clean, separable job. (The redundancy the critics found lives on the K/V side, which is what GQA compresses — last section.)

## The concat-then-W_O structure

```
per head h:  Q^h = x̂ W_Q^h,  K^h = x̂ W_K^h,  V^h = x̂ W_V^h    each (B, S, D_h)
             head_h = softmax(Q^h K^h⊤ / √D_h) V^h              (B, S, D_h)

concat:      Z = [head_1 ; head_2 ; ... ; head_H]               (B, S, H·D_h) = (B, S, D)
mix + write: out = Z W_O                                        W_O ∈ R^(D, D)  →  (B, S, D)
```

The subtle and important part is what `W_O` does. Concatenation just stacks the `H` head outputs side by side; it's `W_O` that **mixes** them and **writes** the result into the residual stream. Partition `W_O` into `H` row-blocks `W_O = [W_O^1 ; ... ; W_O^H]`, each `D_h × D`. Then:

```
out = Z W_O = Σ_h  head_h · W_O^h
```

So the multi-head output is a **sum of per-head writes**, each `head_h W_O^h`. Each head independently writes an update into the stream; `W_O` just names *where* (which directions of `D`) each head writes. This is the read→compute→write frame from file [01](01_qkv_projections.md), now with `H` parallel writers sharing one write port.

**The concat builds the delta, not the stream — the residual update is still pure addition.** A common misread: "heads project the stream *down* to `D_h`, then concat *reconstructs* `D` — so isn't the stream being rebuilt rather than added to?" No. The `(B,S,D)` produced by concat→`W_O` is the attention **output** `out` (the delta), and the actual stream update is:

```
x_new = x + out = x + Σ_h head_h W_O^h
```

The stream `x` is never downscaled and never reconstructed — the down-to-`D_h`, mix, and concat-back-to-`D` all happen on the **branch** (operating on the *copy* `x̂ = Norm(x)`), producing a delta that is *added* to the untouched `x`. Each head reads a `D_h`-dim view and adds back its own rank-≤`D_h` `D`-dim write; the concat is just batched bookkeeping for "sum the heads' contributions." So the identity highway is intact — `x_new = x + (sum of per-head deltas)` — exactly as in Part 3.

**Each head's write is low-rank.** `head_h` is a convex combination of `D_h`-dim value vectors (file [02](02_scaled_dot_product_attention.md)), so `head_h W_O^h` is a `(B,S,D)` update of **rank ≤ D_h = 128** into the `D = 4096` stream. `H` heads sum `H` rank-≤128 updates. This is *more* expressive than one giant head: a single head with `d_head = D = 4096` would be one rank-≤4096 update but with a **single** softmax — still one peak. `H` low-rank heads with `H` softmaxes beat one full-rank head with one softmax, at equal parameter count. (This is the "strictly more expressive... at equal parameter count" claim from the [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md) aside, spelled out.)

**Low-rank means a *subspace*, not reserved coordinate slots — and the next layer is where the writes get remixed.** "Rank ≤ `D_h`" means each head writes into at most `D_h` *directions* of the stream, but `W_O^h` is a full `D_h × D` block, so those directions can be *anywhere* in `R^D` and *which* ones is learned — a head is not assigned a fixed slice of the `D` coordinate axes. The heads write **independently and additively** (no coordination: each just adds `head_h W_O^h` to the shared stream by superposition — [3.1/04](../../part3_residual_connections_deep_networks/3.1_skip_connection/04_residual_stream_as_abstraction.md)). The *combining* is deferred: the **next layer's read-projections** — the following block's `Q/K/V = x̂ W` (or the FFN's `x̂ W_in`) — read the *summed* stream, so each projected output is a linear combination across every direction all the previous heads wrote. That down-projection at the start of the next layer is where the separately-written head contributions get recombined, before that layer's nonlinearity. Write-in-a-direction, read-by-projection, integrate-downstream — the communication protocol of [3.1/04](../../part3_residual_connections_deep_networks/3.1_skip_connection/04_residual_stream_as_abstraction.md), with heads as the writers.

## Why D_h is capped, and why you add heads instead of widening them

`D_h` is held at ~64–128 across model scales — the 8B, 70B, and 405B configs in [ARCHITECTURE.md](../../ARCHITECTURE.md) all use `D_h = 128` and grow `H` with width instead. Two reasons, one from each earlier file:

1. **The SNR / sharpness ceiling ([1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md)).** After `√d_head` scaling the *variance* is fixed, but the *signal-to-noise ratio* of an aligned `q·k` still grows like `√d_head`. Past ~128 the post-softmax distribution gets too sharp — attention collapses toward hard argmax and loses the soft-blending dynamic that makes it trainable and expressive. So `D_h` has a numerics-imposed ceiling.
2. **Hardware.** `D_h = 128` (or 64) is a tensor-core-friendly tile size. Bigger per-head dims don't map as cleanly.

Given a fixed `D`, you then choose: **more heads (smaller each) or fewer heads (bigger each)?** The sharpness ceiling says don't grow `D_h`, and the "one softmax = one peak" argument says you *want* more independent retrievals — so you **add heads**. Width scaling in modern LLMs is `H = D / 128`: 8B has `H=32`, 70B and 405B have `H=64` and `H=128` respectively, all at `D_h=128`.

## Parameter accounting: multi-head is ~free vs. one big head

Because `H · D_h = D` by construction, splitting into heads costs **the same parameters** as one big projection:

| | One head, `d_head = D` | `H` heads, `D_h = D/H` |
|---|---|---|
| `W_Q` total | `D × D` | `H × (D × D_h) = D × D` |
| `W_K`, `W_V` | `D × D` each | `D × D` each |
| `W_O` | `D × D` | `D × D` |
| **QKVO params** | `4D²` | `4D²` |

For `D=4096`: `4 · 4096² ≈ 67M` params either way. Multi-head buys `H` parallel softmaxes and `H` low-rank writes for *zero* extra parameters versus the single-head alternative — it only *rearranges* the same `4D²` budget. (GQA later breaks the K/V symmetry to save inference memory, not params — next section.) Concretely per-block, `W_Q ≈ W_O ≈ 16.8M`, and with GQA `W_K ≈ W_V ≈ 4.2M` — the full table is in [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Head specialization and the interpretability tie-in

Because each head has its own subspace and its own softmax, heads **specialize** — and this is directly observable, not just a story. Named, reproducible head types (forward to Part 11.2):

- **Previous-token heads** — attend from position `t` to `t−1`. Simple, ubiquitous, appear early in training.
- **Induction heads** — the famous one. Pattern: "I've seen `[A][B]` earlier; I'm now at `[A]` again, so attend to and copy `[B]`." They implement in-context copying/pattern-completion and are the mechanistic substrate of a lot of in-context learning. Note this needs the *asymmetric*, *content-vs-position-decoupled* Q/K/V from file [01](01_qkv_projections.md): match on "token after previous `A`" (K), deliver "that token's content" (V).
- **Name-mover heads** — in tasks like indirect-object identification, specific heads move the correct name to the output position.

The point for *this* file: specialization is why `H > 1` pays off *in practice*, and it confirms the "many relations in parallel" argument empirically — you can point at the head doing coreference and the head doing previous-token, in the same layer, on the same token.

## MQA/GQA: evidence that *query-head multiplicity* is the load-bearing part

Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), covered in Part 7.1, share **K and V** across query heads: `H=32` query heads but only `n_kv=8` (or even 1) K/V groups. Quality barely drops. That is a strong clue about *which* multiplicity matters:

- Collapsing the **K/V side** (fewer distinct keys/values) → nearly free.
- But you keep **all the query heads** → because each query head is a separate softmax, a separate "what am I looking for," a separate retrieval. That is the part you can't collapse without hurting.

So the mechanism this whole file describes — *many parallel query softmaxes* — is exactly the part GQA preserves, while the part it compresses (redundant K/V projections) is the part that was never load-bearing. GQA is a natural experiment confirming the "one softmax = one peak, so you need many" thesis. (And it's why file [01](01_qkv_projections.md) framed `W_K, W_V` as the compressible projections.)

## Self-check

1. The [1.2/03](../../part1_math_foundations/1.2_calculus_probability/03_softmax_logsumexp.md) aside insists softmax *sharpness* is not the reason for multiple heads. Then what constrains `D_h`, and what separately motivates `H`? Why don't they collapse into one argument?
2. Multi-head attention with `H` heads of dim `D_h = D/H` has the same `4D²` parameters as one head of dim `D`. So what does the model actually gain from the split — where does the extra expressivity come from if not from parameters?
3. GQA shares K/V across query heads with little quality loss but keeps all query heads. What does that tell you about which part of the multi-head construction is doing the essential work?

### Answers

1. **`D_h` is constrained by softmax sharpness / SNR**: after `√d_head` scaling the aligned-score SNR still grows like `√d_head`, so beyond ~128 the softmax gets too peaky and loses the soft-blend dynamic — a ceiling on per-head *size*. **`H` is motivated by the need for parallel retrievals**: a single softmax is *one normalized selection* — one ranking sharing one mass budget — so it can't run two *independent* selections (by different criteria, with separate budgets) at once, yet a token needs several (prev-token, syntactic head, coreferent, bracket, sink); so you run many softmaxes. They don't collapse because one is about *how wide each softmax can be* (numerics) and the other is about *how many softmaxes you need* (expressivity); you could fix either without the other.
2. `H` **independent softmaxes** and `H` **low-rank writes into the stream**. One head of dim `D` is a single rank-≤`D` write governed by *one* peak-picking softmax; `H` heads are `H` rank-≤`D_h` writes each with *its own* softmax, summed through `W_O`. The gain is combinatorial routing capacity per parameter — many parallel content-based reads instead of one — not more parameters (the count is identical `4D²`).
3. That **query-head multiplicity is the load-bearing part**, and the distinctness of K/V projections is largely redundant. Each query head is a separate retrieval (a separate softmax / "what am I looking for"), which is exactly the thing the one-independent-selection-per-softmax argument says you need many of. Sharing K/V just means those many queries read from a shared set of keys/values — cheaper, and mostly harmless — whereas collapsing the *query* heads would remove the parallel retrievals themselves.

## Exercise

In a notebook, on a toy sequence build two attention layers with **matched parameter count**: (a) a single head with `d_head = D`, and (b) `H = 4` heads with `D_h = D/4`, concatenated and mixed by `W_O`. Confirm the two have the same number of parameters in `W_Q,W_K,W_V,W_O`.

Then construct a task that needs **two retrievals kept separate**: each token's target output carries its **previous token's** value in the first `D/2` dimensions and the **earliest token's** value in the last `D/2` dimensions (the two must land in *different* output dims, not be blended). Hand-set version (b) so head 1 attends to `t−1` and head 2 to position 0, with `W_O` routing head 1 into the first `D/2` dims and head 2 into the last `D/2`; verify it solves the task exactly. Then argue why version (a) can't: a single head applies one `W_V` and produces one *averaged* value vector, so it cannot deliver two different retrieved values into two disjoint output slots. (Note the subtlety: if the target were simply the *average* of the two values, a single head *could* do it by attending 0.5/0.5 — the failure is specifically the need to keep the retrievals *separate* / select by *independent* criteria, not merely to attend to two positions.)
