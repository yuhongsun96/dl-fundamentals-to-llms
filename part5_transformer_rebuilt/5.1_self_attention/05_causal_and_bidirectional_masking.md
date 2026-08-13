# Causal vs. Bidirectional Masking

Attention as built in files [02](02_scaled_dot_product_attention.md)–[04](04_multi_head_attention.md) lets *every* token attend to *every* token. Masking is the one-line modification that restricts *which* tokens a query may see — and that single choice determines whether you get a GPT-style generator or a BERT-style encoder, and whether training is embarrassingly parallel or not. This is where "decoder-only vs. encoder" actually lives.

**Convention:** row-vector (`Y = X W`), repo default. Dims from [NOTATION.md](../../NOTATION.md); 8B config `S ≤ 8192` from [ARCHITECTURE.md](../../ARCHITECTURE.md).

## Causal masking: no peeking at the future

An autoregressive LM factorizes `p(x_1..x_S) = Π_t p(x_t | x_{<t})`. For that to be valid, the representation used to predict `x_t` must depend **only** on tokens `≤ t−1` (equivalently, position `t`'s output may read positions `≤ t`). So we forbid every query from attending to future keys.

Implemented by adding a mask to the scores **before** softmax (recall the `scores` line in file [02](02_scaled_dot_product_attention.md)):

```
scores = Q K^T / √d_k                       (B, H, S, S)
scores = scores + M      where  M[i,j] = 0     if j ≤ i     (past / present: allowed)
                                        −∞    if j > i     (future: forbidden)
A = softmax(scores, axis=keys)              future entries → exp(−∞) = 0
out = A V
```

`M` is upper-triangular (above the diagonal) `−∞`. In fp16/bf16 you use a large negative constant (e.g. `−1e9` or `finfo.min`), not literal `−∞`, to avoid `NaN` from `0 · ∞` — but the effect is identical: after softmax the future weights are exactly 0, so each row `i` is a distribution over keys `≤ i` only. The mask is added *pre*-softmax precisely so the normalization (the `Σ` in softmax) is taken over the *allowed* keys only — masking post-softmax would leave the surviving weights not summing to 1.

The mask is a **fixed function, not learned** ([ARCHITECTURE.md](../../ARCHITECTURE.md) "what's learnable" table) — it's the same triangular matrix every forward pass.

### The payoff: all S predictions, one parallel forward pass

This is the property that made decoder-only training win, and it's worth stating carefully because it's the crux.

With the causal mask in place, a **single** forward pass over the length-`S` sequence produces, at every position `t`, a hidden state that has seen exactly `x_{≤t}` — i.e. the correct context for predicting `x_{t+1}`. So one forward pass yields **all `S` next-token predictions at once**, and you compute the loss for all `S` of them in parallel:

```
one forward pass on (x_1, ..., x_S)  →  logits at all S positions
position t's logits are trained to predict x_{t+1}     (the shifted-by-one target)
loss = mean over t of  CE(logits_t, x_{t+1})           all S terms at once
```

This is **teacher forcing**: during training you feed the *true* prefix (not the model's own samples) at every position, so all positions can be processed simultaneously — there's no sequential dependency in the forward pass, the mask handles causality. Contrast the RNN you know from Part 4: an RNN's hidden state `h_t = f(h_{t−1}, x_t)` is *inherently sequential* — you cannot compute `h_t` before `h_{t−1}`, so you get one loss term per timestep only after `t` serial steps. The Transformer replaces that serial recurrence with a parallel matmul + a static mask. **This parallelism over the sequence dimension is a primary reason Transformers scale on GPUs where RNNs didn't** (Part 4.1's "why they lost").

### The asymmetry: training is parallel, inference is not

The parallel-training win does **not** carry over to generation. At inference you don't have the future tokens (you're producing them), so you decode **one token at a time**: sample `x_{t+1}` from position `t`'s logits, append it, run again. Sequential by nature. Naively that re-attends over the whole growing prefix every step — `O(S²)` total — which is why the **KV cache** (store past `K, V`, only compute the new token's query) exists. Forward: Part 9.2. So causal masking gives you *parallel training, sequential inference* — the reverse of what you might expect, and the source of a lot of modern inference engineering.

## Bidirectional: every token sees every token

Drop the mask entirely. Every query attends to every key, past and future. This is the **encoder** setting — BERT and friends.

```
scores = Q K^T / √d_k         (no mask added)
A = softmax(scores)           every row is a full distribution over all S keys
```

Great for **representations**: a token's encoding is informed by its full left *and* right context, which is what you want for classification, tagging, retrieval embeddings (Part 8.5), extractive QA. But it **destroys the free next-token objective**: if position `t` can already see `x_{t+1}`, then "predict `x_{t+1}`" is trivial (just copy it through). There's no causal factorization to train against.

So bidirectional models need a *different* self-supervised objective — **Masked Language Modeling** (MLM): corrupt ~15% of input tokens with `[MASK]` and train the model to reconstruct them from both-sided context (Part 6.1). MLM is a good representation-learner but a bad *generator*: it predicts a fixed subset of positions per pass rather than a full autoregressive distribution, so you can't cheaply sample text from it, and the `[MASK]` token creates a train/inference mismatch (real inputs have no `[MASK]`).

## Prefix-LM: the hybrid

You can also mask *part* of the sequence. **Prefix-LM** (used by some T5-style and UL2 setups) is **bidirectional over the prompt/prefix, causal over the completion**:

```
prefix (the prompt)          completion (to be generated)
┌─────────────────┐          ┌────────────────────────┐
 tokens attend      each completion token attends to
 both directions    ALL prefix tokens + completion ≤ itself
```

The mask is block-structured: full attention within the prefix block, causal within the completion, and the completion may see the whole prefix. Rationale: the prompt is fully known at inference time, so there's no reason to hide its right context — let the model encode it bidirectionally — while still generating the completion autoregressively.

## The three regimes side by side

| | **Encoder** (bidirectional) | **Decoder** (causal) | **Prefix-LM** |
|---|---|---|---|
| Mask | none | upper-triangular `−∞` | bidirectional on prefix, causal on completion |
| Each token sees | all tokens | tokens `≤ t` | all prefix + completion `≤ t` |
| Enabling objective | **MLM** (fill-in-blank, Part 6.1) | **causal LM** (next-token) | causal LM on completion |
| Training over sequence | parallel (predict masked subset) | **parallel** (predict all `S` shifted) | parallel |
| Free generation? | no (needs MLM, `[MASK]` mismatch) | **yes** (autoregressive) | yes (on completion) |
| Canonical models | BERT, RoBERTa, encoders | GPT, Llama, Mistral | T5-LM adaptation, UL2, PaLM (mixed) |

## Why decoder-only + causal won the scaling race

The full argument is Part 6.1's, but the seed is right here in the masking choice:

- **One objective, every token supervised, in parallel.** Causal LM trains on *all* `S` positions per pass (not a 15% masked subset like MLM), so it extracts more learning signal per token of data and per FLOP — and it does so with the RNN's sequential bottleneck removed.
- **Generation is native.** The training objective (next-token) *is* the inference task. No separate decoder, no `[MASK]` mismatch, no architectural adaptation to generate.
- **Simplicity scales.** One stack, one mask, one loss. Encoder-decoder (T5/BART, Part 6.1) and encoder-only (BERT) each buy something (bidirectional encoding, denoising) but at the cost of the clean "everything is next-token prediction" recipe that turned out to scale the best.

The BERT-era instinct — "bidirectional is obviously better, it sees more context" — is true for *representations* and false for *general-purpose generative modeling at scale*. That reversal is one of the biggest mental-model updates from 2018 to now.

## Self-check

1. The causal mask is added *before* softmax, not applied after. Why does the order matter — what would break if you zeroed out future attention weights *after* the softmax instead?
2. Causal masking lets you train on all `S` next-token predictions in one forward pass, yet generation from the same model is strictly sequential. Explain why the parallelism appears in training but not inference.
3. BERT sees both directions and thus "more context" per token than a causal LM. Why did decoder-only causal models nonetheless win for large-scale generative modeling?

### Answers

1. Softmax normalizes over whatever it's given. Adding `−∞` to future scores *before* softmax removes them from the normalizer, so the surviving (past/present) weights are renormalized to sum to 1 correctly. If you instead softmax over *all* keys and then zero the future weights *after*, the remaining weights no longer sum to 1 (you deleted mass that was in the denominator), so each row is an under-normalized, biased distribution — and gradients would still have flowed through the future logits during the softmax. Pre-softmax masking is what makes the attention distribution a proper distribution over the allowed keys.
2. In **training** you already have the whole true sequence, so teacher forcing feeds the real prefix at every position; the causal mask ensures position `t` only sees `≤ t`, and all `S` positions can be computed in one parallel matmul because there's no data dependency between them (each just reads a masked slice of the same fixed input). In **inference** you're *producing* the future tokens, so position `t+1`'s input literally doesn't exist until you've sampled it from position `t` — an unavoidable serial dependency. The mask enforces causality; only training has the future available to exploit for parallelism.
3. Because the causal next-token objective (a) supervises *every* position every pass rather than a ~15% masked subset, giving more signal per FLOP; (b) *is* the generation task, so no `[MASK]` train/inference mismatch and no separate decoder needed to actually produce text; and (c) is architecturally simpler (one stack, one mask, one loss), which scaled most cleanly. Bidirectional context is genuinely better for *representations* (classification, retrieval), but for scaling a general-purpose *generator* the causal recipe's efficiency and native generation won.

## Exercise

In a notebook, build the causal mask `M` for `S = 6` as a `(6,6)` matrix with `0` on and below the diagonal and `−1e9` above. (1) Add it to a random score matrix, softmax over the last axis, and confirm each row `i` is a distribution supported only on columns `≤ i` (upper-triangle weights are ~0, each row sums to 1). (2) Take a 2-token toy "model," run one *masked* forward pass on a length-6 sequence, and verify that position `t`'s output is unchanged when you overwrite tokens `> t` with garbage — demonstrating that causal masking really does hide the future, which is what makes the single-pass, all-positions training loss valid. (3) Remove the mask and repeat (2): now position `t`'s output *does* change when you edit future tokens — that's the bidirectional regime, and why it can't support a free next-token objective.
