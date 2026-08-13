# Supplementary: SwiGLU — the modern gated FFN

Companion to `../02_matrix_multiplication.md` (Hadamard-product section). The canonical use of elementwise multiplication in modern LLMs. Used by Llama, Mistral, Gemma, DeepSeek, Qwen — essentially every frontier open LM since ~2023.

## The full FFN

```
FFN_SwiGLU(x) = (silu(x W_gate) ⊙ (x W_up)) @ W_down
```

Three matmuls with a Hadamard product in the middle. Replaces the vanilla Transformer FFN `FFN(x) = relu(x W_1) @ W_2`.

## Breaking it down

Input `x ∈ R^(B, S, D)` where `D = d_model`.

### 1. Two parallel up-projections

```
x W_gate:  (B, S, D) @ (D, d_ff) → (B, S, d_ff)      # "gate" branch
x W_up:    (B, S, D) @ (D, d_ff) → (B, S, d_ff)      # "value" branch
```

Two independent Linear layers on the same `x`. Both produce `(B, S, d_ff)`. `d_ff` is the inner dim, typically ~2.67×D (see param-count section).

### 2. SiLU on the gate branch

```
silu(u) = u · sigmoid(u) = u / (1 + e^(-u))
```

Also called Swish with β=1. Smooth, non-monotonic near zero (dips slightly negative around `u = -1`), near-linear for large positive inputs.

```
        silu(u)
  ┌──────────────────────┐
  │           ╱          │
  │         ╱            │
  │      _╱              │
  │   _─                 │
  │──                    │
  └──────────────────────┘
  -4         0         +4
```

Applied elementwise to the gate branch. Shape unchanged: `(B, S, d_ff)`.

### 3. The Hadamard product — the actual gating

```
silu(x W_gate)  ⊙  (x W_up)        # both are (B, S, d_ff)
```

Elementwise multiply. Each of the `d_ff` channels gets its own independent gate (from the gate branch, through SiLU) and its own value (from the up branch, untouched).

**Interpretation**: for each channel, `silu(gate)` decides "how much of the up-branch signal should pass through here." Near 0 → channel suppressed. Positive → channel passes with smooth scaling. Negative → small inversion, since SiLU can go slightly negative.

Unlike ReLU (a hard gate: pass or block), SwiGLU is a **soft learned gate** — the network can learn any smooth per-channel modulation, with both the gating decision and the value being learned functions of `x`.

### 4. Project back down

```
(...) @ W_down:  (B, S, d_ff) @ (d_ff, D) → (B, S, D)
```

Restore `d_model` so the output can be added back to the residual stream.

## The three matrices

| Matrix | Shape | Role |
|---|---|---|
| `W_gate` | `(D, d_ff)` | Pre-activation gate |
| `W_up` | `(D, d_ff)` | Pre-gating value |
| `W_down` | `(d_ff, D)` | Project back to `d_model` |

Vanilla FFN has only `W_1 (D, d_ff)` and `W_2 (d_ff, D)`. SwiGLU adds a third.

## Parameter count — why `d_ff ≈ 2.67 D`

Vanilla FFN with `d_ff = 4D`:
```
params = 2 × D × 4D = 8 D²
```

SwiGLU with `d_ff = X`:
```
params = 3 × D × X = 3 D · X
```

Match vanilla's `8 D²`:
```
3 X = 8 D  →  X = (8/3) D ≈ 2.67 D
```

So Llama uses `d_ff = (2/3) × 4D ≈ 2.67 D`, rounded to a hardware-friendly multiple (128 or 256). **Same total param count as a vanilla FFN** — not a bigger layer, a smarter one.

## Why it replaced the vanilla FFN

Empirical: Shazeer (2020, "GLU Variants Improve Transformer") ran controlled experiments replacing the FFN's nonlinearity with GLU variants at matched parameter count. All GLU variants (SwiGLU, GeGLU, ReGLU, bilinear) gave modest but consistent perplexity improvements. The paper's own conclusion: *"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."*

Despite the cheeky framing, the gain was robust enough that the entire field adopted it. Post-hoc intuitions (not proven):
- **Multiplicative interactions** give more expressive capacity per parameter. `gate(x) × value(x)` composes richer functions than `nonlinear(linear(x))`.
- **Smoother gradients** than ReLU FFNs — SiLU is differentiable everywhere; no dead-neuron problem.
- **Selective channel routing** — each channel can be independently silenced by the gate, yielding learned sparsity without hard zeros.

## GLU family variants

Same gate × value structure, different gate activation:

| Variant | Gate activation |
|---|---|
| GLU (original) | `sigmoid(gate)` |
| GeGLU | `gelu(gate)` |
| **SwiGLU** | `silu(gate)` |
| ReGLU | `relu(gate)` |
| Bilinear | identity (no activation on the gate) |

Performance is broadly similar across variants. SwiGLU and GeGLU dominate modern LLMs.

## Where it lives in the Transformer block

```
                 residual
                    │
               ┌────┴────┐
               │         │
x ──► LayerNorm ── SwiGLU FFN ──┐
         (or RMSNorm)           │
                           add ─┘
                                │
                             next layer
```

Same slot as the classic FFN. Every layer of a Llama-style model does this sequence twice (attention sub-block + FFN sub-block), and the FFN sub-block uses SwiGLU.

## One-line summary

**SwiGLU = "take two parallel linear projections of `x`, apply SiLU to one, multiply them elementwise, project back down."** The multiplication is the whole point — turning the FFN from `nonlinear(linear(x))` into `gate(x) × value(x)`, which is strictly more expressive at matched parameter count.

## Self-check

1. Given `D = 4096`, `d_ff = (8/3) · 4096`, compute the total FFN param count. Compare to a vanilla `d_ff = 4D` FFN.
2. Why does `d_ff = 2.67 D` appear in Llama instead of `4 D`? Could you pick any other value?
3. What's the difference in forward-pass FLOPs between SwiGLU and vanilla FFN at matched params? (Hint: it's about the extra matmul offsetting the smaller `d_ff`.)
4. Why is `silu(x) = x · sigmoid(x)` and not just `sigmoid(x)` the gate activation? What would using `sigmoid` alone buy or cost?

## Exercise

Implement SwiGLU from scratch in PyTorch:
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

Verify the output shape matches the input shape. Count params and confirm `3 × d_model × d_ff`. Compare against a vanilla FFN at matched param count by sweeping `d_ff`.

## Reading

- Shazeer (2020), "GLU Variants Improve Transformer" — short, original ablation.
- PaLM paper (Chowdhery et al., 2022) — first prominent adoption at scale.
- Llama 1/2/3 papers — SwiGLU has been the default FFN ever since.
