# Matrix Multiplication — The Central Operation

## Notation first

Before any of this lands, make sure these symbols are in your working memory. They recur in every DL paper.

### Set and shape notation

- `R` — the real numbers.
- `R^n` — the set of real-valued vectors with `n` components. A vector `x ∈ R^n` has shape `(n,)`.
- `R^(m×n)` — the set of real-valued matrices with `m` rows and `n` columns. A matrix `W ∈ R^(m×n)` has shape `(m, n)`. (Read the superscript `m×n` as "m by n".)
- `R^(m×n×k)` — a real-valued 3D tensor with those dims. Generalizes to any rank.
- `∈` — "is an element of" or "has the type". `W ∈ R^(m×n)` is the math equivalent of a type annotation: *`W` is a real matrix with m rows and n columns.*
- `→` — "maps to". `f: R^n → R^m` means `f` takes an `n`-dim vector and returns an `m`-dim vector.

### The two conventions for "which side is input, which is output"

This trips everyone up. There are two conventions in the wild, and papers switch between them without warning.

**Column-vector convention** (classical linear algebra):
- Vectors are columns. `y = W x`.
- `W ∈ R^(m×n)` maps `R^n → R^m`. **Left dim is output, right dim is input.**
- Matches pure-math textbooks and most derivations.

**Row-vector convention** (most ML papers, because batches of activations are stacked as rows):
- Vectors are rows. `Y = X W`.
- With `X ∈ R^(batch × d_in)` and `W ∈ R^(d_in × d_out)`, output is `R^(batch × d_out)`. **Left dim is input, right dim is output.**
- Matches the "Attention Is All You Need" notation and most Transformer papers.

**Concrete example — two ways to say the same thing:**

A query projection that takes a `d`-dim token and maps it to a `d_k`-dim query vector.
- Column convention: `W_Q ∈ R^(d_k × d)`, applied as `q = W_Q x`.
- Row convention: `W_Q ∈ R^(d × d_k)`, applied as `Q = X W_Q` with `X ∈ R^(n × d)` giving `Q ∈ R^(n × d_k)`.

**These are the same linear map — the matrices are transposes of each other.** Only the notation and the side of the vector differ.

**Which convention is this file using?** The row convention, because that's what you'll see in modern papers (Transformer, BERT, GPT, etc.) and what matches `(B, S, D) @ (D, D_out) → (B, S, D_out)` in PyTorch. When I write `W ∈ R^(d_in × d_out)`, left is input, right is output.

**PyTorch gotcha**: `nn.Linear(in_features=d, out_features=d_k)` stores `weight` with shape `(d_k, d)` (column-convention shape) but applies it as `x @ weight.T` to give row-convention behavior. So `linear.weight.shape[0]` is the output dim, not the input dim. Confusing but widespread.

**Why PyTorch does this**: every layer type stores weights **output-dim-first** — `Linear (out, in)`, `Conv2d (out_ch, in_ch, kH, kW)`, `Embedding (vocab, dim)`. That uniformity means `weight[i]` always means "params for output channel `i`", regardless of layer type. The stored matrix is also the canonical `W` from `y = Wx` (matches paper math and the autograd rule `∂L/∂W = (∂L/∂y) x^T`), and `.T` is free on 2D tensors (just a stride flip — no copy).

**Where the transpose actually happens**: only in `Linear`'s forward. Other layers don't transpose: `Conv2d` uses its own layout, `Embedding` is a row lookup `weight[ids]`, norm params are 1D and broadcast elementwise. Activations and biases never get transposed — they stay in row convention throughout.

**Weight tying** exploits this: the input embedding does `weight[ids]` on a `(V, D)` matrix; the `lm_head` multiplies `hidden @ weight.T` on the same `(V, D)` matrix to produce logits. Same parameters, used in both directions — lookup on the way in, transposed matmul on the way out.

### Dimension-name conventions

The letters used as shape dims aren't random — each community has preferred names:

- **Pure math / generic matmul**: `m, n, k` (or `m, n, p`). Usually: `A ∈ R^(m×k)`, `B ∈ R^(k×n)`, `AB ∈ R^(m×n)`.
- **NLP activations**: `B` (batch), `S` or `T` or `L` (sequence length), `D` or `d_model` (hidden dim), `H` (num heads), `D_h = D/H` (per-head dim), `V` (vocab size).
- **Attention-specific**: `Q, K, V` for query/key/value tensors. `d_k` for key dim, `d_v` for value dim (often `d_k = d_v = D_h`).

When a paper using the row convention says "let `W_Q ∈ R^(d × d_k)`", translate: `W_Q` is a matrix that, applied to a batched input `X ∈ R^(B × S × d)` via `X @ W_Q`, produces a `d_k`-dim query for each token: `R^(B × S × d_k)`.

### Operation notation

- `W @ x`, `Wx`, `W · x` — all mean matrix-vector (or matrix-matrix) multiplication. The `@` is Python/NumPy; the others are math.
- `W^T` (or `Wᵀ`) — transpose. `(W^T)[i, j] = W[j, i]`. Flips rows and columns. If `W ∈ R^(m×n)` then `W^T ∈ R^(n×m)`.
- `<u, v>` or `u · v` or `u^T v` — inner product (dot product) of two vectors. Returns a scalar.
- `W[i, :]` — row `i` of `W`, as a vector. `W[:, j]` — column `j`.
- `‖x‖` — norm of `x` (L2 unless otherwise stated).
- `⊙` — elementwise (Hadamard) product. **Not** the same as matmul.
- `⊗` — outer product (or sometimes Kronecker product — context dependent).

### The superscript `T` gotcha

`W^T` means "transpose". `W^2` means "matrix squared" (i.e., `W @ W`, only defined if `W` is square). `W^(-1)` means "matrix inverse". Don't confuse with element-wise exponentiation — in math papers, `W²` almost never means squaring each entry.

## Two views you need both of

### View 1: Composition of linear maps
A matrix `W ∈ R^(m×n)` is a function `R^n → R^m`. `W @ x` evaluates the function. `W2 @ W1` is function composition. Every neural network (without activations) is one giant matrix — that's why you need nonlinearities.

### View 2: Dot products in parallel
`(W @ X)[i, j] = <W[i, :], X[:, j]>` — each output entry is an inner product. Attention is literally this: Q @ K^T computes every query-key dot product in parallel.

## Matmul's cousins: elementwise and outer products

Matmul is not the only way to multiply tensors. Two other ops show up constantly in modern architectures and do *very different* things.

### Elementwise (Hadamard) product `⊙`

`(A ⊙ B)[i, j] = A[i, j] * B[i, j]`. Same shape in, same shape out. No summing, no mixing of positions.

| Use | Role |
|---|---|
| **Gated FFNs** (SwiGLU, GeGLU) | `SwiGLU(x) = silu(x W_gate) ⊙ (x W_up)` — the gate-value multiplication at the heart of Llama/Gemma/most modern FFNs. Full breakdown: [`supplementary/02_swiglu.md`](supplementary/02_swiglu.md) |
| **LSTM / GRU gates** | Forget/input/output gates scale the cell state per-channel |
| **Dropout** | `x ⊙ mask`, mask is Bernoulli 0/1 |
| **RMSNorm / LayerNorm scale** | `x̂ ⊙ γ` — learned per-feature scale `γ ∈ R^D` broadcasts across batch/sequence |
| **RoPE** | Position encoding as an elementwise rotation: `q ⊙ cos + rotate(q) ⊙ sin` |
| **MoE expert weighting** | Each expert's output scaled by its router probability |
| **Activation backward pass** | `δ ⊙ σ'(pre)` — gradient through a pointwise nonlinearity |

**Why it's everywhere**: it implements **multiplicative gating** — letting or blocking information flow per-channel, independently. Matmul *mixes* features (every output depends on every input via dot product); Hadamard *modulates* features in place.

### Outer product `⊗` (or `u v^T`)

`(u ⊗ v)[i, j] = u[i] * v[j]`. Takes `(m,)` and `(n,)`, produces `(m, n)`. The "opposite" of a dot product: dot collapses two vectors into a scalar; outer expands them into a matrix.

| Use | Where it appears |
|---|---|
| **Weight gradient** | `∂L/∂W = δ x^T`. Every gradient update to a Linear layer's weight is literally an outer product (upstream gradient × input activation), summed across the batch. Happens in every backward pass. |
| **LoRA / low-rank updates** | `ΔW = B A` of rank `r` is a sum of `r` outer products: `ΔW = Σ_i B[:, i] A[i, :]^T`. "Rank-r matrix" literally means "sum of r outer products." |
| **SVD reconstruction** | `A = Σ σ_i u_i v_i^T` — decomposes any matrix into a weighted sum of rank-1 outer products. Basis of low-rank compression, PCA, Eckart-Young approximation. |

### Shape rules compared

| Op | Shape rule | What's summed |
|---|---|---|
| **Dot product** | `(n,) · (n,) → scalar` | The one shared dim collapses to a scalar — maximum reduction |
| **Matmul** | `(m, k) @ (k, n) → (m, n)` | Inner dim `k` summed away; outer dims survive |
| **Outer product** | `(m,) ⊗ (n,) → (m, n)` | **Nothing** summed — matmul with `k = 1`, no real reduction |
| **Elementwise** | `(m, n) ⊙ (m, n) → (m, n)` | Not a matmul — no dim disappears, pure per-position multiply |

The spectrum: dot (everything collapses) → matmul (one dim collapses) → outer (zero collapse, dim created) → Hadamard (no dim manipulation at all).

### Compute profile

- **Matmul**: `O(m · n · k)` FLOPs, compute-bound, uses tensor cores, high arithmetic intensity.
- **Outer product**: `O(m · n)` FLOPs, `O(m + n)` input memory, `O(m · n)` output memory — memory-bound on the output side.
- **Hadamard**: `O(N)` FLOPs, `O(N)` memory — memory-bound. Cheap per-op but usually fused with adjacent ops (e.g. a SwiGLU kernel fuses both matmuls, the silu, and the elementwise mul into one launch to avoid extra memory trips).

### The intuition worth holding

- **Matmul** = "mix all input features into each output feature via weighted sum." How a layer *combines* information.
- **Hadamard** = "modulate each channel's strength without mixing channels." How gates and normalizations *control* information.
- **Outer product** = "two rank-1 directions interacting to define a 2D pattern." Atomic unit of weight updates and low-rank structure.

You need all three to read modern architectures: SwiGLU is Hadamard, LoRA is outer products, attention scores and projections are matmul.

## Shape rules

```
(m, k) @ (k, n) → (m, n)
```

The inner dims must match and disappear; the outer dims survive.

## Batched matmul

PyTorch's `@` / `torch.matmul` treats all but the last two dims as batch dims and broadcasts:

```
(B, H, S, D_h) @ (B, H, D_h, S) → (B, H, S, S)   # attention scores (per-head)
(B, S, D) @ (D, V) → (B, S, V)               # output projection / lm_head
```

The last example is what happens at the final layer of every LM: project from `d_model` to `vocab_size`. For a 50K vocab and `d_model=4096`, that's a 200M-parameter matrix — often the single largest in the model (shared with the input embedding via **weight tying** — one `(V, D)` matrix serves both as the input lookup and the output logit projection, so the model learns a single token↔vector mapping instead of two; common in smaller LMs where these params are a large fraction of the total, mostly dropped at frontier scale where the redundancy is affordable).

## Why it's so fast on GPUs

Matmul is **compute-bound** for reasonable shapes: O(mnk) FLOPs with O(mn + nk + mk) memory. The arithmetic intensity makes it ideal for tensor cores. This is why every DL op that can be expressed as matmul is expressed as matmul.

Inference of small-batch LLMs is the exception — it becomes **memory-bound** because you're loading huge weight matrices to multiply by a tiny activation. Keep this in mind for Part 9 (inference).

## Associativity and why it matters

`(A @ B) @ C = A @ (B @ C)` — but the cost is different.

```
A: (1000, 10), B: (10, 1000), C: (1000, 5)

(A @ B) @ C: forms 1000×1000 intermediate → 10M + 5M = 15M ops, big memory
A @ (B @ C): forms 10×5 intermediate → 50K + 50K = 100K ops, tiny memory
```

### LoRA — the canonical application

LoRA (Low-Rank Adaptation) is this associativity trick applied to fine-tuning. Fine-tuning a pretrained Linear layer means finding a weight update `ΔW ∈ R^(d × d)`. Storing and applying a full `d × d` delta is prohibitive — for `d = 4096` that's 16.7M extra params per layer, per adapter, and a full-size matmul on every forward pass. LoRA parameterizes the delta as a rank-`r` product:

```
ΔW = B @ A,   B ∈ R^(d × r),  A ∈ R^(r × d),  r ≪ d    (typically r = 8–64)
```

At inference, the naive path would be:
```
output = (W + BA) @ x            # compute the d×d delta, then apply
```
But this forms the full `(d, d)` product `BA` on every forward pass. Associativity lets you reorder:

```
(W + BA) @ x  =  W @ x  +  (B @ A) @ x  =  W @ x  +  B @ (A @ x)
                           ───────┬───────    ──────┬──────
                           forms a d×d           never forms d×d
                           intermediate          intermediate (only R^r)
```

Comparing the two paths (ignoring the `W @ x` term, which is the same in both):

| Path | Shape sequence | FLOPs | Peak intermediate |
|---|---|---|---|
| `(B @ A) @ x` | `(d, r) @ (r, d) → (d, d) → (d,)` | `d²r + d²` | `(d, d)` matrix |
| `B @ (A @ x)` | `(r, d) @ (d,) → (r,) → (d,)` | `2 r d` | `(r,)` vector |

For `d = 4096, r = 16`:
- Left path: `d²r + d² = 268M + 16.7M ≈ 285M` ops, forms a 16.7M-entry intermediate.
- Right path: `2rd = 131K` ops, forms a 16-entry intermediate.
- **~2200× fewer FLOPs, no giant intermediate.**

This is exactly the same trick as the tall-skinny / short-wide example above — always prefer the ordering that keeps intermediates small. LoRA just applies it to the specific case where the "skinny middle" is the low-rank bottleneck of a learned weight delta.

**Training vs inference**:
- **Training**: `W` is frozen; only `A` and `B` receive gradients. Forward is `W @ x + B @ (A @ x)` — the un-merged form. The small intermediate keeps memory and optimizer state tiny, which is why LoRA fine-tuning fits on consumer GPUs.
- **Inference**: once you're done training, you can compute `W_merged = W + B @ A` once and apply it as a single matmul — same speed as the base model. This sacrifices per-query adapter-swapping but eliminates any overhead. Serving many adapters at once → keep un-merged and pay the small extra cost per query; single-adapter deployment → merge.

For the broader rank-`r` story (initialization, why `ΔW` is low-rank in the first place, other places low-rank shows up), see `04_outer_products_low_rank.md`.

## Self-check

1. For `(B, S, D) @ (D, 4D)` (the FFN up-projection), what's the FLOP count? The output memory?
2. Why is weight tying (sharing the embedding matrix with `lm_head`) a meaningful param savings?
3. In `softmax(QK^T / √d) @ V`, what are the two matmuls and what's the intermediate shape?

### Answers

1. **FLOPs**: `2 · B · S · D · 4D = 8 · B · S · D²` (counting fused multiply-add as 2 ops). For `B=8, S=2048, D=4096`: ~2.2 TFLOPs per layer. **Output memory**: `B · S · 4D` floats. In bf16 (2 bytes): `B · S · 4D · 2 = 8 · B · S · D` bytes. For the same shapes: ~537 MB just for one FFN's intermediate activation.
2. The `(V, D)` matrix is huge — for `V=50K, D=4096` it's 200M params, often the single largest matrix in the model. Without tying you have **two** such matrices (input embedding + `lm_head`), doubling that already-massive cost. Tying them saves `V · D` params per model, often 5–30% of total. Common in smaller LMs (GPT-2, BERT). Frontier models often skip it because the absolute cost is affordable and untied params allow input/output specialization.
3. Two matmuls. **Score**: `Q @ K^T → (B, H, S, S)` (then divided by `√d` and softmax'd). **Output**: `softmax(...) @ V → (B, H, S, D_h)`. The intermediate `(S, S)` attention matrix is the **memory bottleneck** — quadratic in sequence length. FlashAttention's central trick is to never materialize this full S×S tensor: it tiles the computation and recomputes during backward to keep memory linear in S.

## Exercise

Implement attention scoring by hand with for-loops, then replace it with one `@` call. Confirm identical output. You should feel viscerally why the vectorized form is the only sane choice.
