# Notation

A single-page reference for symbols and conventions used across the curriculum. Skim it once, then come back when something looks unfamiliar. Entries flagged **(forward)** are not yet introduced in detail but will appear in later parts — they're listed here so the symbol isn't a surprise.

If the same symbol is used differently in different parts, both meanings are listed and the file disambiguates.

---

## Conventions

These are repo-wide defaults, but matrix convention varies by file — see the per-file note at the top of each.

| Convention | Default | Notes |
|---|---|---|
| Matrix shapes | **row-vector** in most files | Activations as rows: `Y = X W`, with `(B, S, d_in) @ (d_in, d_out) → (B, S, d_out)`. Matches modern Transformer papers and PyTorch shapes. Derivation files in 1.1/1.2 use column-vector — see below. |
| Logarithm base | **natural log (nats)** for loss math | Convert to bits / perplexity in eval contexts only. `log₂(e) ≈ 1.4427` is the nats→bits factor. |
| Index origin | **1-based** in math (`x_1, ..., x_n`) | Code may use 0-based. The repo's prose uses 1-based to match standard math notation. |
| Gradient shape | **same as parameter** | `∂L/∂W` is always shape of `W`. This is the deepest invariant in autograd. |
| Vector orientation | column by default in math | A "vector" `x ∈ R^n` is a column unless written `x^T`. Display-wise, prose may write entries on a row to save space. |

## Dimension names

Standard letters used as tensor dimension labels. Use these when they apply; fall back to `m, n, k` only for generic matmul examples.

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `S` (or `T`, `L`) | Sequence length (tokens) |
| `D` (or `d_model`) | Model hidden / residual-stream dimension |
| `H` | Number of attention heads |
| `D_h` (or `d_head`) | Per-head dimension, `= D / H` |
| `d_k`, `d_v` | Key / value dimensions (usually `= D_h`) |
| `d_ff` | FFN hidden dimension (typ. `4D`, or `~2.7D` for SwiGLU) |
| `V` | Vocabulary size |
| `d_in`, `d_out` | Input / output dims of a generic linear layer |
| `r` | Rank (LoRA, low-rank decomposition) |
| `m, n, k, p` | Generic matrix dimensions (when none of the above apply) |

---

## Sets, spaces, and quantifiers

| Symbol | Meaning |
|---|---|
| `R` | The real numbers |
| `R^n` | `n`-dimensional real vector space |
| `R^(m×n)` | Real `m × n` matrices |
| `R^(m×n×k)` | Real 3-index tensors |
| `Z`, `N` | Integers, naturals |
| `∈` | "is an element of" / "has type" |
| `f: A → B` | `f` maps elements of `A` to elements of `B` |
| `↦` | "maps to" for individual elements (`x ↦ f(x)`) |
| `∀`, `∃` | "for all", "there exists" |
| `iff` | if and only if |

---

## Linear algebra

### Vectors and matrices

| Symbol | Meaning |
|---|---|
| `x`, `y` (lowercase) | Vectors. Column by default. |
| `W`, `A` (uppercase) | Matrices |
| `X`, `Y` (uppercase) | Batched activations / data matrices (rows are samples) |
| `x_i` | `i`-th entry of vector `x` |
| `A_{ij}` or `A[i, j]` | `(i, j)` entry of matrix `A` |
| `A[i, :]`, `A[:, j]` | Row `i` / column `j` of `A` (NumPy/PyTorch slice notation) |
| `x^T`, `A^T` | Transpose |
| `A^(-1)` | Matrix inverse |
| `I` (or `I_n`) | Identity matrix (of size `n`) |
| `0` | Zero vector / matrix (shape from context) |
| `diag(v)` | Diagonal matrix with `v` on the diagonal |
| `diag(A)` | Vector of diagonal entries of `A` |
| `trace(A)` | Sum of diagonal entries |
| `det(A)`, `|det(A)|` | Determinant; absolute value used in normalizing-flow log-likelihoods **(forward)** |
| `rank(A)` | Rank |

### Products

| Symbol | Meaning |
|---|---|
| `A B` or `A @ B` | Matrix multiplication (no symbol = matmul) |
| `A · B` | Matmul (when readability needs an explicit dot) |
| `⟨u, v⟩` or `u · v` or `u^T v` | Inner / dot product → scalar |
| `u v^T` or `u ⊗ v` | Outer product → matrix (rank-1) |
| `u ⊙ v` (Hadamard) | Element-wise product |
| `u ⊕ v` | Element-wise sum (rare; usually just `+`) |

### Norms

| Symbol | Meaning |
|---|---|
| `‖x‖` or `‖x‖_2` | L2 (Euclidean) norm, `√(Σ x_i²)` |
| `‖x‖_1` | L1 (Manhattan) norm, `Σ |x_i|` |
| `‖x‖_∞` | L∞ (max) norm, `max_i |x_i|` |
| `‖x‖_p` | Lp norm, `(Σ |x_i|^p)^(1/p)` |
| `‖A‖_F` | Frobenius norm — Euclidean treating `A` as a flat vector |
| `‖A‖_2` or `σ_max(A)` | Spectral norm (largest singular value) |
| `‖A‖_*` | Nuclear norm (sum of singular values) |

### Eigendecomposition and SVD

| Symbol | Meaning |
|---|---|
| `λ`, `v` in `A v = λ v` | Eigenvalue, eigenvector |
| `A = V Λ V^(-1)` | Eigendecomposition (`Λ` = diagonal of eigenvalues, `V` = eigenvector matrix) |
| `A = U Σ V^T` | Singular value decomposition |
| `σ_i` | `i`-th singular value (note clash with sigmoid below — context disambiguates) |
| `u_i`, `v_i` | Left / right singular vectors |
| `κ(A) = σ_max / σ_min` | Condition number |
| `ρ(A)` | Spectral radius (max `|λ_i|`) |

### Projections and orthogonality

| Symbol | Meaning |
|---|---|
| `u ⊥ v` | Orthogonal: `⟨u, v⟩ = 0` |
| `proj_u(x)` | Vector projection of `x` onto the direction of `u` |
| `Q^T Q = I` | Defining condition for an orthogonal matrix `Q` |
| `SO(n)` | Special orthogonal group (orthogonal + `det = +1`) — the rotations of `R^n` |
| `R(θ)` | `2 × 2` rotation block `[[cos θ, −sin θ], [sin θ, cos θ]]`; the atom of the canonical form — 1.1/07 |
| `G(i, j, θ)` | Givens rotation: identity except an `R(θ)` block at rows/cols `i, j` (spins one plane, fixes the rest) — 1.1/07 |

---

## Calculus and differentiation

### Derivatives

| Symbol | Meaning |
|---|---|
| `df/dx` (straight `d`) | Total derivative — `f` is a function of one variable |
| `∂f/∂x_i` (curly `∂`) | Partial derivative — `f` has many inputs, vary `x_i`, hold others fixed |
| `∂/∂x_i` | The partial-derivative operator (`∂f/∂x_i` is this operator applied to `f`) |
| `f'(x)`, `f''(x)` | 1D derivatives (first, second) |
| `∇f` ("nabla", "del", "grad f") | Gradient of scalar-valued `f`: vector of all partials, shape `(n,)` |
| `∇²f` or `H_f` | Hessian: matrix of second partials, shape `(n, n)` |
| `D_u f(x)` | Directional derivative along unit vector `u` |

`∂f` alone is **not** a meaningful object — `∂` always pairs with a `/∂x_i` to specify the variable. Read `∂f/∂x_i` as one symbol, not as a literal fraction.

### Jacobians and matrix differentiation

| Symbol | Meaning |
|---|---|
| `J` or `J_f(x)` | Jacobian of `f: R^n → R^m` at `x`, shape `(m, n)`, `J[i,j] = ∂f_i/∂x_j` |
| `∇_x L` | Gradient of scalar `L` w.r.t. vector `x`, shape `(n,)`. Equals `J^T` for scalar `f`. |
| `∂L/∂W` | Gradient of scalar loss w.r.t. matrix parameter `W`, **same shape as `W`** |
| `J δ` | Jacobian-vector product (**JVP**, forward-mode) |
| `v^T J` or `J^T v` | Vector-Jacobian product (**VJP**, reverse-mode) — backbone of backprop |
| `diag(σ'(x))` | Jacobian of an element-wise activation `σ` |

### Chain rule

| Form | Statement |
|---|---|
| Scalar: `h = g ∘ f` | `h'(x) = g'(f(x)) · f'(x)` |
| Multivariate (matrix form) | `J_h = J_g · J_f` (matmul, not elementwise) |
| Backprop (gradient form) | `∇_x L = J_f^T · ∇_{f(x)} L` |
| Linear layer (column-vector): `y = Wx + b` | `∂L/∂W = (∂L/∂y) x^T`, `∂L/∂x = W^T (∂L/∂y)`, `∂L/∂b = ∂L/∂y` |
| Linear layer (row-vector): `Y = X W + b` | `∂L/∂W = X^T (∂L/∂Y)`, `∂L/∂X = (∂L/∂Y) W^T`, `∂L/∂b = sum over batch of ∂L/∂Y` |

---

## Probability

### Random variables and distributions

| Symbol | Meaning |
|---|---|
| `X ~ p` or `X ~ p(x)` | `X` is a random variable distributed as `p` |
| `X ~ N(μ, σ²)` | Gaussian with mean `μ` and variance `σ²` |
| `X ~ Uniform(a, b)` | Uniform on `[a, b]` |
| `g ~ Gumbel(0, 1)` | Gumbel noise (used for categorical sampling, Gumbel-softmax) |
| `p(x)` | PMF or PDF of `x` (context decides) |
| `p(x, y)` | Joint distribution |
| `p(y | x)` | Conditional distribution |
| `p_θ(x)` | Distribution with parameters `θ` |
| `X ⊥ Y` | `X` and `Y` are independent |
| `X ⊥ Y | Z` | `X` and `Y` are conditionally independent given `Z` |

### Expectations and moments

| Symbol | Meaning |
|---|---|
| `E[X]` or `E_{x~p}[f(x)]` | Expectation (subscript names the distribution when ambiguous) |
| `Var(X)` | Variance |
| `Cov(X, Y)` | Covariance |
| `E[X | Y]` | Conditional expectation |
| `∫ f(x) p(x) dx` | Continuous expectation written out |

### Reparameterization

| Symbol | Meaning |
|---|---|
| `x = g_θ(ε)`, `ε ~ p_0` | Reparameterized sample (push noise through a parametric map) |
| `∇_θ E_{x~p_θ}[f(x)]` | Gradient of expectation w.r.t. distribution parameters |
| `∇_θ log p_θ(x)` | Score function (REINFORCE / policy gradient) |

---

## Information theory

All entropies are in **nats** (natural log) unless flagged as bits.

| Symbol | Meaning |
|---|---|
| `H(p)` or `H(X)` | Entropy: `-Σ p(x) log p(x)` (or integral) |
| `H(p, q)` | Cross-entropy: `-Σ p(x) log q(x)` |
| `H(X | Y)` | Conditional entropy |
| `KL(p ‖ q)` | KL divergence: `Σ p(x) log(p(x)/q(x))` — forward, mean-seeking |
| `KL(q ‖ p)` | Reverse KL — mode-seeking (used in RL / variational inference) |
| `I(X; Y)` | Mutual information: `KL(p(x,y) ‖ p(x) p(y))` |
| `NLL` | Negative log-likelihood, `-log p_θ(x)` (per-sample) |
| `PPL` | Perplexity, `exp(mean NLL in nats)` |
| `BPB` | Bits per byte: `loss_nats × log₂(e) / bytes_per_token` |
| `nats`, `bits` | Units (natural log vs. log base 2). `1 nat = log₂(e) ≈ 1.4427 bits` |
| `-log p_i` | Information content / "surprise" of outcome `i` |

---

## Activations and special functions

| Symbol | Meaning |
|---|---|
| `exp(x)`, `e^x` | Exponential |
| `log(x)` | Natural log (unless explicitly `log_2`, `log_10`) |
| `log_2(x)` | Base-2 logarithm |
| `√x` | Square root |
| `σ(x)` | Sigmoid `1/(1+e^{-x})` (**not** a singular value here — context decides) |
| `tanh(x)` | Hyperbolic tangent |
| `ReLU(x) = max(0, x)` | Rectified linear unit |
| `GELU(x)` | Gaussian error linear unit |
| `silu(x)` or `swish(x)` | `x · σ(x)` |
| `softmax(x)_i = e^{x_i} / Σ_j e^{x_j}` | Softmax |
| `log_softmax(x)_i = x_i - logsumexp(x)` | Log-softmax (numerically stable) |
| `logsumexp(x) = log Σ_i e^{x_i}` | LSE — stable form of `log(Σ exp)` |
| `δ_{ij}` | Kronecker delta: 1 if `i = j`, else 0 |
| `1[condition]` | Indicator: 1 if true, 0 else |
| `clip(x, a, b)` | Clamp to `[a, b]` |
| `‖x‖_2` Jacobian: `x^T / ‖x‖` | Listed in 1.1/08 Jacobian table |

---

## Deep learning / Transformer notation

### Activations and layers

| Symbol | Meaning |
|---|---|
| `L` | **Scalar loss.** Function of weights `W` (data fixed) or of input `x` (when computing input gradients). See "What `L` is a function of" in [supplementary/08](part1_math_foundations/1.1_linear_algebra/supplementary/08_linear_layer_gradients.md). |
| `ℓ(ŷ, y)` | Per-sample loss |
| `θ` | Generic model parameters (all weights and biases bundled) |
| `f_θ(x)` | Model `f` with parameters `θ` applied to input `x` |
| `ŷ`, `y_pred` | Model prediction **(forward — generation)** |
| `logits` | Raw pre-softmax scores **(forward)** |
| `h_l` | Hidden state at layer `l` (residual stream) **(forward)** |
| `h + f(h)` | Residual block: stream `h` plus sublayer output `f(h)` (the skip connection) — Part 3.1 |
| `α_l`, `diag(λ)` | Per-layer residual-branch scale — scalar (ReZero) or per-channel (LayerScale) — Part 3.2 |
| `embedding(ids)` | Token-embedding lookup |
| `W_E ∈ R^(V, D)` | Token embedding matrix |
| `W_U` or `W_E^T` | Unembedding / output projection (often tied to `W_E`) |

### Attention **(forward — Part 5)**

| Symbol | Meaning |
|---|---|
| `Q`, `K`, `V` | Query, Key, Value matrices |
| `W_Q`, `W_K`, `W_V` | Projection weights producing `Q`, `K`, `V` from the input stream |
| `W_O` | Output projection (after attention) |
| `Q K^T / √d_k` | Scaled dot-product attention scores |
| `α` or `attention_weights` | Softmaxed attention scores, shape `(B, H, S, S)` |
| `softmax(Q K^T / √d_k) V` | Single-head attention output |
| `head_i` | `i`-th attention head |
| `cos(q, k)` or `cos_sim` | Cosine similarity (used in retrieval / contrastive losses) |
| `RoPE(q, k, pos)` | Rotary position embedding applied to `q`, `k` **(forward)** |
| `pos`, `pos_idx` | Position index in a sequence |

### Adapters and fine-tuning **(forward — Part 7)**

| Symbol | Meaning |
|---|---|
| `ΔW = B A` | LoRA weight update (low-rank), `B ∈ R^(D, r)`, `A ∈ R^(r, D)` |
| `W_merged = W + α/r · B A` | Merged weight after LoRA fine-tune (with scaling factor) |
| `r` | LoRA rank |

### RL / RLHF **(forward — Part 8)**

| Symbol | Meaning |
|---|---|
| `π_θ(a | s)` | Policy distribution over actions given state |
| `π_ref` | Reference / frozen policy (KL anchor) |
| `Q(s, a)` | Action-value function |
| `V(s)` | State-value function |
| `r`, `r(s, a)` | Reward |
| `β · KL(π_θ ‖ π_ref)` | KL penalty term in PPO / DPO objectives |
| `A(s, a)` | Advantage function |

### Sampling and generation **(forward — Part 9)**

| Symbol | Meaning |
|---|---|
| `T`, `τ` | Temperature (note: `T` also = sequence length elsewhere; context disambiguates) |
| `top-k`, `top-p` | Truncation parameters for decoding |
| `softmax_T(x)_i = exp(x_i/T) / Σ exp(x_j/T)` | Temperature-scaled softmax |

### Embeddings / retrieval **(forward — Part 8.5)**

| Symbol | Meaning |
|---|---|
| `pool(H)` | Pooling: token states `(S, D)` → one vector `(D,)` (CLS / mean / last-token / learned) |
| `e` | A pooled, L2-normalized text embedding, `‖e‖ = 1` |
| `(q, p)` | Query–passage (or anchor–positive) contrastive pair |

### Losses

| Symbol | Meaning |
|---|---|
| `CE(p, q)` | Cross-entropy loss |
| `-log q_{y}` | Per-token CE for the true class `y` |
| `L_InfoNCE` | Contrastive (CLIP-style) loss |
| `L_DPO`, `L_PPO` | Preference / policy losses **(forward)** |

---

## Symbol collisions worth knowing

A few symbols are reused with different meanings. Context always disambiguates, but watch for these:

| Symbol | Possible meanings |
|---|---|
| `σ` | Sigmoid activation **or** singular value `σ_i` **or** standard deviation `σ` in `N(μ, σ²)` |
| `T` | Sequence length **or** temperature **or** matrix transpose `^T` |
| `L` | Scalar loss **or** layer count **or** sequence length (rare; prefer `S`) |
| `H` | Entropy `H(p)` **or** number of attention heads **or** Hessian `H_f` |
| `D` | Model dimension **or** directional derivative `D_u f` **or** dataset `D` **or** dataset size in *tokens* (scaling-law convention, Part 6.3 — there the model width is always written `d_model`) |
| `N` | Parameter count (scaling laws, Part 6.3: `C ≈ 6ND` with `C` = training FLOPs) **or** batch/sample count **or** the naturals |
| `α` | Attention weights **or** LoRA scale **or** generic step size / hyperparameter **or** per-layer residual-branch scale (ReZero `α`, DeepNorm `α=(2L)^{1/4}`) — Part 3.2 |
| `β` | KL coefficient **or** generic hyperparameter |
| `V` | Vocab size **or** Value matrix in attention **or** matrix of right singular vectors in SVD |
| `r` | LoRA rank **or** reward **or** generic radius |
| `J` | Jacobian **or** rarely a loss name (avoid) |
| `i` | Generic index / position **or** the imaginary unit `√−1`. Watch for this in RoPE: the matrix derivation uses `i`, `j` for positions, but the complex-number view switches positions to `m`, `n` so `i` can be `√−1` — Part 5.3/04 |

When in doubt, the file using the symbol will define it at first use.

---

## How to extend this file

When you introduce a symbol that isn't here:

1. Add a row to the most relevant table above.
2. If it'll be reused across many files, mention it; if it's local to one file, define it inline there and skip this file.
3. If it collides with an existing symbol, add it to the **collisions** table.
4. Keep entries one line where possible — this is a quick-reference, not a tutorial. Tutorials live in the curriculum files.
