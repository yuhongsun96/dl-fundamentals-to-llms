# Q, K, V: Three Learned Views of the Residual Stream

Self-attention starts by taking the one thing every token has — its residual-stream vector — and projecting it three different ways. That's the whole setup. The subtlety is entirely in *why three* separate projections, and what each one is for. Get that right and scaled dot-product attention (file [02](02_scaled_dot_product_attention.md)) is almost mechanical.

**Convention:** row-vector (`Y = X W`), the repo default — activations are rows, so `(B, S, D) @ (D, d_out) → (B, S, d_out)`. Dimension names from [NOTATION.md](../../NOTATION.md): `B` batch, `S` sequence, `D = d_model` hidden, `H` heads, `D_h = D/H` per-head dim, `d_k = D_h` key dim. Numbers anchor to the 8B config in [ARCHITECTURE.md](../../ARCHITECTURE.md): `D=4096, H=32, D_h=128`.

## The read → compute → write frame

Everything a Transformer block does is a cycle on the residual stream (Part 3.1, [residual stream](../../part1_math_foundations/1.1_linear_algebra/supplementary/06_residual_stream.md)):

```
read   :  x̂ = RMSNorm(x)          normalized copy of the stream   (B, S, D)
compute:  attn = Attention(x̂)      the mixing happens here          (B, S, D)
write  :  x = x + attn             add back into the raw stream     (B, S, D)
```

Attention is the *compute* step. Its job: let each position pull information from other positions. The Q/K/V projections are how a token declares **what it wants** (query), **what it can be matched on** (key), and **what it will hand over if matched** (value). Expanding the *compute* box so nothing is a black box (full mechanics in files [02](02_scaled_dot_product_attention.md), [04](04_multi_head_attention.md)):

```
Q,K,V   = x̂ W_Q, x̂ W_K, x̂ W_V        (B, S, D_h)   project the stream three ways
scores  = Q Kᵀ / √d_k                  (B, S, S)     who matches whom
A       = softmax(scores)              (B, S, S)     attention weights
head_out= A · V                        (B, S, D_h)   weighted blend of values (in the head's workspace)
attn    = head_out · W_O               (B, S, D)     W_O: project back up to stream width, then write
```

Two quick reminders on that box before we return to the projections. **The `/√d_k`** on the scores is the *scaled* in "scaled dot-product attention": `q·k` sums `d_k` terms so its variance grows ~`d_k`, and dividing by `√d_k` keeps logits ~unit-scale — without it, larger `d_k` pushes softmax into its saturated, near-zero-gradient regime (full treatment, file [02](02_scaled_dot_product_attention.md)). It's a *per-op, head-width* fix, unrelated to depth. And `d_k = D_h` here by the standard construction (`d_k = d_v = D_h = D/H`), so `√d_k` and `√D_h` are the same number — they're only distinct concepts because `q,k` *must* share the dot-product dim `d_k` while the value dim `d_v` is in principle free (and GQA/MLA, Part 7.1, do decouple them).

The last line is the piece to hold onto for section 3: **`W_O` (per head, `∈ R^(D_h, D)`) is the output projection** — the final linear map that takes the head's `D_h`-dim result and maps it back up to the `D`-dim model width so it can be added into the residual stream (`x = x + attn`). (In the packed multi-head form you concatenate all `H` heads to `(B,S,D)` and apply one `W_O ∈ R^(D,D)` — same thing, per file [04](04_multi_head_attention.md).) So `W_V` is where content *enters* the head from the stream and `W_O` is where it *leaves* the head back into the stream.

## The three projections

For one head, from the normalized input `x̂ ∈ (B, S, D)`:

```
Q = x̂ W_Q      W_Q ∈ R^(D, D_h)     →  (B, S, D_h)     "what each token is looking for"
K = x̂ W_K      W_K ∈ R^(D, D_h)     →  (B, S, D_h)     "what each token offers as a match target"
V = x̂ W_V      W_V ∈ R^(D, D_h)     →  (B, S, D_h)     "what each token contributes if matched"
```

Three linear maps, no nonlinearity, no bias (modern LLMs drop biases — [2.1/03](../../part2_neural_network_fundamentals/2.1_mlp_building_block/03_bias_terms.md)). In practice all `H` heads' projections are packed into one `D × (H·D_h) = 4096 × 4096` matrix and reshaped to `(B, S, H, D_h)`, but conceptually each head owns its own `W_Q^h, W_K^h, W_V^h`. Per-head treatment is file [04](04_multi_head_attention.md).

The dating-app analogy, because it sticks: every token writes a **query** ("looking for: the subject of my clause"), advertises a **key** ("offering: I am a subject noun"), and keeps a **value** ("if you pick me, here's my content"). Q and K decide *who talks to whom*; V is *what actually gets said*.

## Why three separate matrices

This is the question the file exists to answer. Three decouplings, each load-bearing.

### 1. Query ≠ key: "what I search with" ≠ "what I'm found by"

If you forced `W_Q = W_K`, matching would be *symmetric*: token A attends to B exactly as much as B attends to A (scores `x̂_A W W^T x̂_B^T` would be a symmetric form).

*What "symmetric" means here, plainly:* the score matrix reads the same when flipped across its main diagonal — entry `[i, j]` equals entry `[j, i]` for every pair. Concretely, that means **the score for A→B is forced to equal the score for B→A**: A and B must find each other *equally compatible*. (The forced-equal quantity is the raw pre-softmax score; the final weights can differ slightly because softmax normalizes each row by its own sum — but the model can no longer make A find B relevant while B finds A irrelevant.)

But linguistic relations are directional. A pronoun looks *back* to its antecedent; the antecedent is not looking for the pronoun. A verb seeks its subject; the subject isn't seeking the verb the same way. Separate `W_Q` and `W_K` let the score matrix be **asymmetric** — A→B and B→A are independent. That asymmetry is most of what attention patterns encode.

### 2. Match space ≠ content space: (Q·K) ≠ V

`W_Q, W_K` define the space where *matching* happens (the dot product `q·k`). `W_V` defines the space of *content that moves*. Tying value to key would force "what I'm matched on" to equal "what I deliver" — but a token is often retrieved for one reason and read for another. Example: an induction head (file [04](04_multi_head_attention.md), Part 11.2) matches on *"the token after a previous occurrence of the current token"* (a positional/identity signal in K) but delivers *the actual next-token content* (in V). Different subspaces, so different matrices.

### 3. Read (W_V) vs. write (W_O) are the stream I/O

`W_V` reads content *out* of the stream into the head's workspace; `W_O` (defined above — the output projection) writes the head's output *back* into the stream. These are the two points where attention touches the residual stream's content. `W_Q, W_K` never do: they *read* `x̂` too, but they collapse it to **scalar scores** (`Q Kᵀ`), and those scores only become the *weights* `A` — they decide *how much* to read from each position, never contributing a content vector themselves. So of the four matrices, `W_V`/`W_O` carry content (the payload), `W_Q`/`W_K` set the routing (the weights) — see below.

## W_Q, W_K only matter as a product

Here is the sharp fact worth internalizing. `W_Q` and `W_K` appear **only** inside the score:

```
scores = Q K^T = (x̂ W_Q)(x̂ W_K)^T = x̂ (W_Q W_K^T) x̂^T
```

The entire attention pattern depends on `W_Q, W_K` **only through the product `W_Q W_K^T ∈ R^(D, D)`** — a single `D × D` bilinear form. Insert any invertible `M ∈ R^(D_h × D_h)`: replace `W_Q → W_Q M` and `W_K → W_K M^{-T}` and the product `W_Q M M^{-1} W_K^T = W_Q W_K^T` is unchanged. So the individual `W_Q, W_K` are **not identifiable** — there's a whole gauge orbit of matrices giving identical attention. The interpretable object is the combined **QK circuit** `W_Q W_K^T`, not either matrix alone (this is exactly how mechanistic-interpretability work factors attention — Part 11.2).

Contrast the value path. `W_V` and `W_O` compose into the **OV circuit** `W_V W_O ∈ R^(D, D)`, which is *what content gets written where* in the stream. So an attention head is really two low-rank `D × D` maps: a QK circuit (where to look, rank ≤ `D_h`) and an OV circuit (what to move, rank ≤ `D_h`). Four matrices, two circuits.

| Matrices | Compose into | Governs | Touches stream content? |
|---|---|---|---|
| `W_Q, W_K` | QK circuit `W_Q W_K^T` (`D×D`) | *Where* each token attends | No — only sets the score |
| `W_V, W_O` | OV circuit `W_V W_O` (`D×D`) | *What* content is moved | Yes — read (`W_V`) and write (`W_O`) |

## Why not fewer matrices

- **One shared matrix** (`W_Q = W_K = W_V`): kills both decouplings above — symmetric matching *and* match-space = content-space. You could barely express directional relations or "retrieve for one reason, read for another." Strictly less expressive at the same param count.
- **Two matrices** (tie K and V, or tie Q and K): recovers one decoupling but not the other. Tying K=V is closest to workable and shows up in some retrieval setups, but the standard three-matrix form dominates because the extra matrix is cheap (attention is only ~20% of block params — [ARCHITECTURE.md](../../ARCHITECTURE.md)) and buys real expressivity.

Historical note: the original 2017 Transformer already used three separate projections. What *changed* since BERT is not the count but the layout — heads share K/V groups now (GQA, Part 7.1), which collapses the K/V *side* while keeping query heads plural. That's evidence (developed in file [04](04_multi_head_attention.md)) that query multiplicity is the load-bearing part, and the K/V projections are the compressible ones.

## Self-check

1. You tie `W_Q = W_K`. What structural property does the score matrix `Q K^T` acquire, and name one linguistic relation it can no longer represent well.
2. Someone claims "head 7 in layer 3 uses this specific `W_K`." Why is that statement not well-defined, and what *is* the well-defined object?
3. Why is it fine to tie `W_V` and `W_K` in principle harder to justify than tying, say, nothing — i.e. what does the value projection need to do that the key projection doesn't?

### Answers

1. It becomes **symmetric**: `scores = x̂ (W W^T) x̂^T`, and `W W^T` is symmetric PSD, so `score(A→B) = score(B→A)` for all pairs. You lose **directional/asymmetric relations** — e.g. a pronoun attending back to its antecedent (the antecedent shouldn't attend forward to the pronoun with equal weight), or a verb attending to its subject. Attention patterns are overwhelmingly asymmetric, so this is a large loss.
2. `W_Q` and `W_K` enter the model *only* through the product `W_Q W_K^T`. Any invertible `M` gives `(W_Q M)(W_K M^{-T})^T = W_Q W_K^T`, so infinitely many `(W_Q, W_K)` pairs produce identical behavior — the individual matrix is a gauge-dependent, non-identifiable quantity. The well-defined, interpretable object is the **QK circuit** `W_Q W_K^T` (equivalently the attention pattern it induces).
3. `W_V` defines the **content space** — the actual information copied into the destination token's stream (read by `W_V`, written by `W_O`). `W_K` defines only the **match space** — it never contributes content, only decides *whether* a match fires. A token is frequently retrieved for a reason (its key) unrelated to what it should deliver (its value): induction heads match on "token identity for copying" but deliver "the following token's content." Forcing K = V would collapse those two roles, so keeping `W_V` separate is doing real work that a shared K/V cannot.

## Exercise

In a notebook, take `x̂` of shape `(B, S, D) = (1, 4, 8)` (one sequence, 4 tokens, tiny `D`) with `D_h = 8`, `H = 1`. Initialize `W_Q, W_K, W_V` as random `8×8` matrices.

1. Compute `Q, K, V` and the raw score matrix `S_raw = Q K^T` (shape `(4, 4)`). Confirm it is **not** symmetric.
2. Now set `W_K = W_Q` and recompute. Confirm the score matrix is now symmetric (`S_raw ≈ S_raw^T` to float tolerance).
3. Back to independent `W_Q, W_K`. Draw a random invertible `M ∈ R^(8×8)`, set `W_Q' = W_Q M` and `W_K' = W_K (M^{-1})^T`, and confirm `Q' K'^T ≈ Q K^T`. You've just demonstrated the QK-circuit gauge freedom by hand — the individual projections changed, the attention pattern didn't.

### Solution

Row-vector convention (`Q = x̂ W_Q`), so `x̂` is `(S, D) = (4, 8)`, each `W ∈ R^(8×8)`, and `scores = Q Kᵀ` is `(4, 4)`.

```python
import numpy as np
np.random.seed(0)

S, D, D_h = 4, 8, 8
x = np.random.randn(S, D)                 # x̂, one sequence of 4 tokens
W_Q, W_K, W_V = (np.random.randn(D, D_h) for _ in range(3))

def scores(Wq, Wk):
    return (x @ Wq) @ (x @ Wk).T          # Q Kᵀ, shape (4, 4)

# 1) independent W_Q, W_K  → asymmetric
S1 = scores(W_Q, W_K)
print(np.abs(S1 - S1.T).max())            # 19.49  → NOT symmetric

# 2) tie W_K = W_Q         → symmetric
S2 = scores(W_Q, W_Q)
print(np.abs(S2 - S2.T).max())            # 0.0    → symmetric exactly

# 3) gauge transform       → attention pattern unchanged
M   = np.random.randn(D_h, D_h)           # det ≈ 51.5, invertible
Wq2 = W_Q @ M
Wk2 = W_K @ np.linalg.inv(M).T
S3  = scores(Wq2, Wk2)
print(np.abs((x @ Wq2) - (x @ W_Q)).max())  # 17.94  → the projection really changed
print(np.abs(S3 - S1).max())                # 1.1e-13 → scores identical
```

**Part 1 — independent `W_Q, W_K` give an asymmetric score matrix.** With `seed=0`:

```
Q Kᵀ =
[[-48.461 -16.597 -29.261 -42.414]
 [-12.003  -4.586  -8.531  -3.401]
 [-48.754 -18.957   4.020 -25.266]
 [-61.256  -9.735 -16.968 -41.172]]
max |S − Sᵀ| = 19.49        # e.g. S[0,3] = −42.41 but S[3,0] = −61.26
```

`score(i→j) ≠ score(j→i)` — token `i` attending to `j` is independent of `j` attending to `i`. This asymmetry is what lets attention encode *directional* relations (a pronoun attends back to its antecedent without the antecedent attending forward equally).

**Part 2 — tying `W_K = W_Q` forces symmetry.** Now `scores = x̂ (W_Q W_Qᵀ) x̂ᵀ`, and `W_Q W_Qᵀ` is symmetric, so the whole form is:

```
max |S − Sᵀ| = 0.00        # symmetric exactly (a bilinear form with a symmetric middle matrix)
```

The score matrix collapses to symmetric — `score(i→j) = score(j→i)` for every pair — which is exactly the expressivity you lose by tying the two projections.

**Part 3 — the QK circuit is gauge-invariant.** With random invertible `M` (`det ≈ 51.5`):

```
max |Q' − Q|        = 17.94     # the individual projection Q changed a lot
max |Q'K'ᵀ − Q Kᵀ|  = 1.1e-13   # ...but the attention scores are identical (float noise)
```

Because `Q'K'ᵀ = (x̂ W_Q M)(x̂ W_K M⁻ᵀ)ᵀ = x̂ W_Q (M M⁻¹) W_Kᵀ x̂ᵀ = x̂ W_Q W_Kᵀ x̂ᵀ`, the `M` cancels. The individual `W_Q, W_K` (and `Q, K`) moved substantially, yet the attention pattern is bit-for-bit unchanged — so the identifiable object is the product `W_Q W_Kᵀ` (the QK circuit), not either matrix alone. This is the gauge freedom from the "`W_Q, W_K` only matter as a product" section, demonstrated numerically.
