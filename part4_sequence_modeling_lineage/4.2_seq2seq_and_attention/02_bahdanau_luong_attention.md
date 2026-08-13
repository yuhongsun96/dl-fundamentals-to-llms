# Bahdanau/Luong Attention as Learned Alignment

This is the pivot of the whole lineage. Bahdanau et al. (2015) dissolved the seq2seq bottleneck (file `01`) with one idea — let the decoder look at *all* encoder states, weighted by relevance — and in doing so wrote down **query–key–value attention** two years before the Transformer named it. Read this file as "attention, first contact": everything in Part 5 is a refinement of what's here.

**Convention:** column-vector; encoder states `s_1…s_S ∈ R^D`; decoder state at step `t` is `d_t ∈ R^D`.

## The mechanism

At each decoder step `t`, instead of reading a single fixed `c`, compute a **fresh context vector** `c_t` as a weighted average of all encoder states:

```
e_{t,j} = score(d_t, s_j)                 alignment score: how relevant is source pos j to decoder step t?
α_{t,j} = softmax_j(e_{t,j})              normalize over all source positions → weights summing to 1
c_t     = Σ_j α_{t,j} · s_j               weighted sum of encoder states → the context for this step
```

Then the decoder uses `c_t` (which now *changes every step*) to produce `y_t`. The scoring function is where Bahdanau and Luong differ:

- **Bahdanau (2015), "additive" attention:** `score(d, s) = vᵀ tanh(W_d d + W_s s)` — a small learned MLP. Computed *before* the decoder state update, and the whole thing is trained end-to-end with no alignment supervision.
- **Luong (2015), "multiplicative" attention:** `score(d, s) = dᵀ W s` (general) or just `dᵀ s` (dot). Cheaper, and — crucially — the **dot-product** form is the one the Transformer keeps. Luong also introduced *global* vs *local* attention (attend to all vs. a window) and simpler plumbing.

The `α_{t,j}` are a **soft alignment**: for translation, plotting the `T × S` matrix of weights recovers a near-diagonal word-alignment map (English "the cat" ↔ French "le chat"), *learned* without ever being told the alignment. Attention arrived not as "a memory mechanism" but as **differentiable, learned alignment** — the model deciding, per output step, which inputs to read.

## This is query–key–value in embryo

Rename the pieces and Part 5 falls out:

| Bahdanau/Luong (2015) | Transformer QKV (2017) | role |
|---|---|---|
| decoder state `d_t` | **query** `q` | "what am I looking for?" |
| encoder states `s_j` | **keys** `k_j` *and* **values** `v_j` | "what's available to match / to read" |
| `score(d_t, s_j)` | `qᵀk / √d_k` | compatibility |
| `α_{t,j} = softmax_j` | `softmax(qᵀK/√d_k)` | normalized weights |
| `c_t = Σ α s_j` | `Σ α v_j` = attention output | weighted read |

It is the **soft dictionary lookup** from the [NOTATION attention section](../../NOTATION.md): a query retrieves a blend of values, weighted by how well the query matches each key. What the Transformer adds on top (Part 5.1) is a set of specific upgrades, each a fix to a limitation visible right here:

- **Separate keys and values** (Bahdanau uses `s_j` as both) — decouples "what you match on" from "what you retrieve."
- **Scaled dot-product** `/√d_k` — dot scores grow with dimension and saturate the softmax; the scale fixes it (the [attention-logit-growth](../../ARCHITECTURE.md) concern, Part 5.1).
- **Multi-head** — one alignment per step is unimodal; several heads attend to several things at once.
- **Self-attention** — Bahdanau's attention is *cross*-attention (decoder→encoder); pointing attention at a sequence *itself* is the leap that makes recurrence removable (file `03`).

## Why it matters in modern LLM work

- **It's the actual origin of the mechanism the field is built on.** Scaled dot-product attention (Part 5), cross-attention in VLMs (Part 10.3), retrieval-as-attention — all are this 2015 idea, generalized. The "attention as learned soft lookup" mental model you carry through the whole curriculum starts on this page.
- **Alignment interpretability starts here.** The `α` matrix as an inspectable alignment is the ancestor of attention-map analysis and the interpretability work in Part 11.2 (induction heads are a specific, powerful alignment pattern).
- **It reframes what the bottleneck fix *was*.** Not "a bigger context vector" but "**stop compressing; keep everything and select**." That principle — direct, weighted access over a stored set — is precisely what the Transformer scaled up and what the KV cache (Part 9.2) is the machinery for.

## Self-check

1. Write the three-line attention computation (`e`, `α`, `c_t`) and say, for each line, what it does in one phrase.
2. Vanilla seq2seq's context `c` was the same at every decoder step; Bahdanau's `c_t` is not. Why is "recomputed per step" the thing that kills the bottleneck?
3. Map decoder state, encoder states, and the score function onto query, key/value, and compatibility. Which one thing does the Transformer *split* that Bahdanau kept merged?
4. Bahdanau attention is powerful but is *cross*-attention over an RNN encoder–decoder. What single generalization turns it into the ingredient that lets you delete the RNN entirely (file `03`)?

### Answers

1. `e_{t,j} = score(d_t, s_j)` — score how relevant each source position is to this decoder step; `α_{t,j} = softmax_j(e_{t,j})` — turn scores into weights that sum to 1 over source positions; `c_t = Σ_j α_{t,j} s_j` — read a weighted blend of encoder states as this step's context.
2. Because a per-step context lets the decoder attend to *different* source positions at different output steps (and to *all* of them, undamped), so no single fixed vector has to carry the whole source at once — the compression bottleneck is gone; capacity now scales with `S` (all `s_j` are available), not fixed at `D`.
3. `d_t` = query, `s_j` = key *and* value, `score` = compatibility, `softmax` = weights, `Σα s_j` = output. The Transformer **splits keys from values** (Bahdanau uses the same `s_j` for both), decoupling what you match against from what you retrieve.
4. **Self-attention** — pointing the query/key/value construction at a *single* sequence (each position attends to the others) rather than decoder-attending-to-encoder. With that, a stack of attention layers can build representations without any recurrence, which is the Transformer (file `03`).

## Exercise

(a) For a source of length `S = 4` and a single decoder step, write out `c_t = Σ_j α_{t,j} s_j` with concrete weights `α = [0.7, 0.2, 0.05, 0.05]` — which source position dominates the context, and what would the weights look like for a "monotonic" translation at the first target word? (b) Replace `score(d,s) = vᵀtanh(W_d d + W_s s)` with `score = dᵀs` and state what you lose and gain (cost, expressiveness). (c) In two sentences, explain why plotting the full `T×S` matrix of `α` values gives an *alignment* and why that made attention feel interpretable in a way RNN hidden states never were.
