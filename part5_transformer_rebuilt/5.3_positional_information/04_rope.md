# RoPE — Rotary Position Embedding

This is the modern default. Every open LLM you'll touch — Llama, Mistral, Qwen, DeepSeek, Gemma — encodes position with **RoPE** (Su et al., 2021), and [ARCHITECTURE.md](../../ARCHITECTURE.md) forward-references this file as the deep dive for its component #5. RoPE is the synthesis the previous files were building toward: it is **relative** like T5/ALiBi ([file 03](03_relative_positions.md)), **parameter-free** like ALiBi, injected **inside attention** like both — but instead of *adding* a distance bias to the score, it **rotates** the query and key vectors, so the relative property falls directly out of the dot product's geometry. No addition, no table, no bias term. Just a rotation.

**Convention:** row-vector (`Y = X W`). RoPE acts on the per-head query/key vectors `q, k ∈ R^(D_h)` — the 8B config has `D_h = 128`, so each `q`/`k` splits into `D_h/2 = 64` coordinate pairs. Dimensions from [NOTATION.md](../../NOTATION.md); `H = 32`, `S` up to 8192 from [ARCHITECTURE.md](../../ARCHITECTURE.md). For the rotation/orthogonal-matrix facts leaned on below, see [projections and orthogonality](../../part1_math_foundations/1.1_linear_algebra/06_projections_orthogonality.md).

## The core idea in one sentence

**Rotate `q` and `k` by an angle proportional to their absolute position, and the dot product `q · k` comes out depending only on the difference of positions.** You inject *absolute* position (each vector is rotated by an angle tied to its own index) but attention *reads* relative position (the score sees only the offset). That is the whole trick, and it is worth internalizing before any algebra: rotation is the one operation where composing two of them subtracts their angles inside an inner product.

## Why rotation gives you relative-from-absolute

Work in 2D first — RoPE is just this, done `D_h/2` times in parallel. Let `R(θ)` be the 2D rotation matrix (see [orthogonality](../../part1_math_foundations/1.1_linear_algebra/06_projections_orthogonality.md) for why it's orthogonal, `R(θ)^T R(θ) = I`):

```
R(θ) = [ cos θ   −sin θ ]
       [ sin θ    cos θ ]
```

Rotate the query at position `i` by `iθ` and the key at position `j` by `jθ`. Their dot product is:

```
(R(iθ) q) · (R(jθ) k) = q^T R(iθ)^T R(jθ) k
                      = q^T R(−iθ) R(jθ) k              # transpose of a rotation is the inverse rotation
                      = q^T R((j − i)θ) k               # rotations compose by adding angles
```

The absolute angles `iθ` and `jθ` **cancel**, leaving only `(j − i)θ` — a function of the **relative offset**. This is the payoff:

- Feed in absolute positions (`i`, `j`), get out a score that depends only on `j − i`.
- **Zero parameters** — `θ` is a fixed schedule, `R` is deterministic.
- Because `R` is **orthogonal, `‖R(iθ) q‖ = ‖q‖`** exactly — rotating cannot inflate or shrink `q` or `k`, so it never destabilizes the logits. Position is encoded purely as a *phase*, not a magnitude.

Compare to file 03: T5/ALiBi *add* `b(i−j)` to the score after computing `q·k`. RoPE bakes the `(i−j)` dependence *into* `q·k` itself. It's multiplicative/geometric, not additive.

## The complex-number view (why it's clean)

> **Notation.** This section writes positions as **`m`** (query) and **`n`** (key), because `i` is now the **imaginary unit**. Elsewhere in the file positions are `i`, `j`.

**Step 1 — a complex number is just a 2D point.** `z = a + b i` *is* the point `(a, b)`: real part is the x-coordinate, imaginary part is the y-coordinate. Nothing mystical — it's a container for two numbers. So "pair up the `D_h` coordinates into `D_h/2` complex numbers" only regroups the same data: `(q_0, q_1)` becomes one point `z_0 = q_0 + q_1 i`, `(q_2, q_3)` becomes `z_1`, and so on — 64 points in 64 planes for `D_h = 128`.

**Step 2 — multiplying by `e^{iθ}` *is* the rotation matrix.** Euler's formula gives `e^{iθ} = cos θ + i sin θ`, which as a point is `(cos θ, sin θ)` — the unit-circle point at angle `θ`, the same object the rotation matrix is built from ([1.1/07](../../part1_math_foundations/1.1_linear_algebra/07_rotations.md); for *why* Euler's formula holds, see [supplementary/07](../../part1_math_foundations/1.1_linear_algebra/supplementary/07_eulers_formula.md)). Multiply it out:

```
(a + bi)(cos θ + i sin θ)  =  (a cos θ − b sin θ)  +  i (a sin θ + b cos θ)
                              └───── new x ─────┘       └───── new y ─────┘
```

Those two components are *exactly* the rows of `R(θ) · (a, b)ᵀ`. Same arithmetic, one symbol instead of a matmul. The general reason: complex multiplication **adds angles and multiplies magnitudes**, so `r e^{iφ} · e^{iθ} = r e^{i(φ+θ)}` — magnitude untouched, angle bumped by `θ`. (Check: rotating `(1,0)` by 90° is `1 · i = i`, the point `(0,1)`. The x-axis became the y-axis.) So RoPE on one pair is simply:

```
RoPE(z_d, pos) = z_d · e^{i · pos · θ_d}
```

**Step 3 — recovering a dot product needs the conjugate.** Plain `z_q · z_k` is *not* the dot product; complex multiplication mixes the components the wrong way. The correct recipe is multiply by the **conjugate** (`conj(z_k) = k_0 − k_1 i`, which flips the sign of the imaginary part) and keep the **real part**:

```
z_q · conj(z_k) = (q_0 + q_1 i)(k_0 − k_1 i)
                = (q_0 k_0 + q_1 k_1)   +   i (q_1 k_0 − q_0 k_1)
                  └─ the dot product ─┘       └─ cross product, discarded ─┘
```

That's why `Re[...]` and `conj(...)` appear — they're the machinery for pulling a real dot product out of complex arithmetic. The reason it works: conjugating flips a point's angle, so multiplying by a conjugate **subtracts** angles, giving `Re[z_q conj(z_k)] = |z_q| |z_k| cos(φ_q − φ_k)` — magnitude times magnitude times the cosine of the angle between them, which is precisely what a dot product *is*.

**Step 4 — now put the positions in.** Query at position `m`, key at position `n`, each rotated by its own absolute angle:

```
Re[ (z_q e^{i m θ_d}) · conj(z_k e^{i n θ_d}) ]
  = Re[ z_q conj(z_k) · e^{i m θ_d} · e^{−i n θ_d} ]      # conj(e^{i n θ}) = e^{−i n θ}
  = Re[ z_q conj(z_k) · e^{i (m − n) θ_d} ]               # exponents add
```

The absolute positions **cancel**; only the offset `m − n` survives in the phase. Same result as the matrix derivation above — but here the only fact used was "exponents add when you multiply."

**Why this is the cleaner formulation.** Both derivations use the same facts in different clothing:

| Concept | Matrix form | Complex form |
|---|---|---|
| Rotate | `R(θ) v` (2×2 matmul) | `z · e^{iθ}` (one multiply) |
| Angles add | `R(α)R(β) = R(α+β)` (a theorem) | `e^{iα} e^{iβ} = e^{i(α+β)}` (exponent rule) |
| Inverse / transpose | `R(θ)ᵀ = R(−θ)` (needs orthogonality) | `conj(e^{iθ}) = e^{−iθ}` (flip a sign) |
| Dot product | `qᵀ k` | `Re[z_q conj(z_k)]` |
| Extracting the offset | needs `(AB)ᵀ = Bᵀ Aᵀ` plus two rotation lemmas | exponents add — done |

In the complex world "rotate" is a single multiply and the angle bookkeeping happens automatically in the exponent. That's why real implementations often cast `q`/`k` to complex, multiply by a precomputed `e^{i · pos · θ}` table, and cast back — fewer operations, and the relative property is self-evident.

## The frequency schedule: a positional clock

One rotation angle for the whole vector would be a single clock hand — it wraps, and can't distinguish `i` from `i + period`. RoPE instead gives **each coordinate pair its own frequency**, borrowing the frequency-ladder idea from sinusoidal encodings ([file 02](02_sinusoidal_and_learned_absolute.md)):

```
θ_d = base^(−2d / D_h)      d = 0, 1, ..., D_h/2 − 1      (base typically 10000)
```

- **Pair `d = 0`**: `θ_0 = base^0 = 1` rad per step — the fastest hand, spins quickly, resolves adjacent tokens but wraps fast.
- **Pair `d = D_h/2 − 1`**: `θ ≈ base^{−1}` — the slowest hand, drifts almost imperceptibly across the whole sequence, resolves coarse "early vs. late."
- Everything in between spins at a geometrically-spaced rate.

The ladder at `D_h = 128`, base 10000 (so `d = 0…63`, and **`d` indexes a pair**, not a single dimension):

| `d` | dims | `θ_d` (rad/token) | cycle length |
|---|---|---|---|
| 0 | 0, 1 | 1.000000 | **6 tokens** |
| 8 | 16, 17 | 0.316228 | 20 tokens |
| 16 | 32, 33 | 0.100000 | 63 tokens |
| 32 | 64, 65 | 0.010000 | 628 tokens |
| 48 | 96, 97 | 0.001000 | 6,283 tokens |
| 63 | 126, 127 | 0.000115 | **54,410 tokens** |

Consecutive pairs differ by a constant factor `base^(2/D_h) = 1.1548` — each step 15.5% slower — and the ladder crosses one order of magnitude every 16 pairs. Note the slowest cycle: **54,410 tokens with base 10000.** Past roughly that context length *every* pair has wrapped at least once, leaving no unambiguous coarse position signal anywhere in the head — which is the failure [file 05](05_context_length_extension.md) exists to fix.

Together the `D_h/2` pairs form a **positional clock**: fast hands pin down fine position, slow hands the coarse position, and the combination gives every offset a distinct phase signature across the 64 pairs. It's the sinusoidal ladder from file 02 — but applied *multiplicatively to `q` and `k` inside attention*, not additively to the input embedding.

## The score is a filter bank — and the only knob is amplitude

Everything about how RoPE behaves is readable off one exact identity. Split `q` and `k` into their 64 pairs; the rotated score is a **sum of independent channels**, one per pair:

```
qᵀ R(Δ) k  =  Σ_d  ‖q_d‖ ‖k_d‖ · cos(θ_d Δ + φ_d)
```

where `Δ = j − i` is the offset and `φ_d` is the angle between that pair's `q` and `k` components before any rotation. Each channel oscillates in `Δ` at its own fixed frequency; its **amplitude** is `‖q_d‖ ‖k_d‖`. Two things follow, and the second corrects a tempting misconception.

**The knob the model has: amplitude.** `‖q_d‖` and `‖k_d‖` are set by `W_Q` and `W_K`, which sit *upstream* of the rotation. A head that writes near-zero energy into a pair multiplies that channel's oscillation by ~0 — the term drops out of the sum. It's a product, so zeroing *either* side suffices. Trained models show exactly this specialization: heads doing local/syntactic matching load the fast channels; heads doing long-range semantic matching load the slow ones; different heads weight the bands differently.

**The knob it doesn't have: direction.** You might hope a head could dodge a fast channel by pointing its `q`/`k` somewhere "orthogonal to the rotation." It can't. Inside a rotating 2-D plane there is no safe direction — a rotation moves *every* nonzero vector in the plane — and RoPE rotates **all 64 planes simultaneously**, so no direction of head space is left fixed. (The high-dimensional "almost everything is nearly orthogonal" intuition doesn't apply: the planes are fixed and axis-aligned, and the rotation acts inescapably plane by plane.) Escape is by *gain*, never by *geometry*.

So the picture to keep: **RoPE partitions the head dimension into ~64 frequency channels, and each head does per-channel gain control over a fixed filter bank.** What never changes is the norm — `‖Rv‖ = ‖v‖` — so impact is always phase, never magnitude. (And rotating `q` and `k` by the *same* angle changes nothing: `⟨Rq, Rk⟩ = ⟨q, k⟩`. Sensitivity only appears under a *relative* rotation — exactly the `Δθ_d` above.)

**The ladder makes the channels wildly unequal.** For the *same* offset, phase accumulated differs by four orders of magnitude:

| Pair | `θ_d` | Phase after `Δ = 100` tokens | Behavior |
|---|---|---|---|
| Fastest (`d = 0`) | `1.0` | ~100 rad — many full turns | Decorrelates almost immediately; resolves *local* offsets |
| Middle (`d ≈ 32`) | `~0.01` | ~1 rad | Meaningful medium-range signal |
| Slowest (`d = 63`) | `~10⁻⁴` | ~0.01 rad — barely moved | Nearly position-invariant across the whole context |

The rule that compresses the whole table: **each channel serves offsets comparable to (or shorter than) its own cycle.** Within its cycle a channel's phase is unambiguous and its value stable; past it, the channel wraps and its contribution flips sign unpredictably as the offset varies. So a head matching content at `Δ ≈ 1000` runs on the channels whose cycles are ≥ thousands of tokens and gain-suppresses the rest — asking the 63-token channel about `Δ = 1000` is asking a local-syntax instrument a long-range question. (The precise criterion is about the head's *offset tolerance*: a channel is usable when `θ_d × (range of offsets the head must treat alike) ≪ π`. A wrapped channel isn't dead — at a *fixed* offset its phase is still deterministic, and the fast channels are exactly what make a sharp "attend ~`k` back" detector sharp — but for matching across a band of offsets, only sub-cycle channels stay coherent.)

Two consequences:

- **Long-range attenuation is dilution, not a penalty.** At large `Δ`, a fast channel's `cos(θ_d Δ + φ_d)` is effectively a random value in `[−1, 1]` — zero-mean, not systematically negative (measured: mean ≈ 0, std ≈ 0.65 across the fast band). A head spreading energy over many channels sees its long-range score shrink toward zero because terms *cancel*, not because anything pushes it down. Contrast ALiBi ([file 03](03_relative_positions.md)), which really does subtract a growing penalty. RoPE's much-cited "long-term decay" is this dephasing envelope.
- **Context extension has to be per-band.** This is precisely why NTK-aware scaling and YaRN beat uniform Position Interpolation ([file 05](05_context_length_extension.md)): PI squashes every channel equally and damages the fast ones carrying local resolution, while NTK/YaRN protect the fast end and stretch the slow end. That design only makes sense because the channels are unequal.

### The frequencies are fixed; the *allocation* is learned

`θ_d` is a hand-designed schedule with zero parameters, and learnable-frequency variants have shown only modest gains. That's less surprising in the filter-bank picture: **the fixed ladder is a menu of clock speeds, and learning chooses which features read which clock.** `W_Q`/`W_K` cannot invent a rate that isn't on the menu, but they can route any feature to any rate — or blend across several. That is why fixing the frequencies costs so little expressivity, the same argument as coordinate-aligned rotation planes being no real restriction ([1.1/07](../../part1_math_foundations/1.1_linear_algebra/07_rotations.md)).

## Concrete shapes (8B config, `D_h = 128`)

```
q, k              : (B, H, S, D_h)         = (B, 32, S, 128)   per-head queries/keys
split into pairs  : (B, H, S, D_h/2, 2)    = (..., 64, 2)      64 coordinate pairs
θ                 : (D_h/2,)               = (64,)             one frequency per pair, fixed
angles = pos·θ    : (S, D_h/2)             = (S, 64)           per position, per pair
                                                               → cos, sin caches, each (S, 64)
RoPE(q), RoPE(k)  : (B, H, S, D_h)                             same shape in as out (it's a rotation)
```

The `cos`/`sin` tables are precomputed once for all positions `0..S−1` and reused every layer. In practice RoPE is applied per head to `q` and `k` right before the `q·kᵀ/√D_h` score, at every layer.

## Applied to q and k only — not to v, not in the stream

Two placement facts that matter:

- **Only `q` and `k` are rotated, never `v`.** RoPE exists to shape the *attention scores* (which come from `q·k`); the values `v` are the content being retrieved and carry no positional rotation. Rotating `v` would corrupt the payload for no benefit.
- **Nothing is added to the [residual stream](../5.1_self_attention/01_qkv_projections.md).** Unlike absolute encodings (file 02), RoPE injects no positional vector at the input. Position is applied fresh, inside each attention layer, to the projected `q`/`k` only — so it never competes with content for stream channels and never has to "survive" 32 layers of writes. This is exactly the "position injected inside attention" bullet from [ARCHITECTURE.md](../../ARCHITECTURE.md) component #5.

## Why RoPE beat everything

Line the properties up against files 02–03:

| Property | Sinusoidal | Learned abs | T5 bias | ALiBi | **RoPE** |
|---|---|---|---|---|---|
| Relative? | latent only | no | yes | yes | **yes (exact, from the dot product)** |
| Parameters | 0 | `max_S·D` | small table | 0 | **0** |
| Where injected | input | input | logit bias | logit bias | **rotate q,k in attention** |
| Touches residual stream? | yes | yes | no | no | **no** |
| Modulates full q/k interaction? | — | — | no (scalar add) | no (scalar add) | **yes (rotates the vectors)** |
| Extrapolation | mediocre | none | limited | strong | **good, degrades late (→ file 05)** |

RoPE is the one scheme that is relative *and* parameter-free *and* in-attention *and* lets position rotate the full `q`/`k` geometry (rather than just offsetting the score by a scalar, as ALiBi/T5 do) — and it **composes with the dot product for free**: the relative property isn't an added term, it's an algebraic identity of rotations. That combination is why it displaced everything else.

## Where it fails (and the bridge to file 05)

RoPE extrapolates *better* than absolute encodings, but it has **two distinct failure modes** — one about training coverage, one structural — and they're worth separating because the fixes differ.

### Failure 1 — past the trained length: unseen phase

During training the model only sees positions up to `S_train` (say 8192), so each frequency's phase `pos · θ_d` covers a bounded range. At inference past `S_train`, the **fast pairs** spin into phase values the model **never observed** — attention for those channels degrades, and quality falls off past the training length. (The slow pairs are fine; they barely moved even at `S_train`.) This is the failure that Position Interpolation, NTK-aware scaling, and YaRN repair — [file 05](05_context_length_extension.md).

### Failure 2 — past the slowest cycle: no channel left to invest in

The gain-control story above has a hidden requirement: **there must be some channel that is still sub-cycle over the distances you care about.** A head wanting near-position-invariant matching (attend to the system prompt or tool definitions from anywhere) parks its energy in the slow channels. But at base 10000 the slowest cycle is **54,410 tokens** — so at `Δ = 1M` even the slowest channel has wrapped ~18 times, and the 64 phases `θ_d Δ mod 2π` are, for practical purposes, pseudo-random. Then:

- The channel sum becomes a sum of **random-sign terms**: expectation ≈ 0, typical size only `~1/√(2m)` of the fully-aligned score. A strong semantic match at that distance produces a *small logit of unpredictable sign*.
- **Training cannot learn around this**, even though the rotation is deterministic. The offset to the prefix *varies with the query's position* — a query at 700k and one at 900k see completely different phase configurations for the same key. Attending to token 0 from *everywhere* would need the oscillating sum to be large at essentially every `Δ`, which is impossible at nonzero amplitude. Deterministic phase is only exploitable at **fixed** offsets, and "the system prompt" is not at a fixed offset from anywhere.

In this regime, direct content-based attention to the start of context is **structurally unavailable — not hard, unavailable**. What survives is indirect, and it's why degradation is gradual rather than a cliff: **attention sinks** (softmax is relative, so mass drains to early tokens by default — but that's positional garbage-collection, not reading the prompt's content) and **multi-hop relay** (prefix information carried forward through layers in shorter, sub-cycle hops — lossy and low-bandwidth).

### The consequence: the base is a precondition, not a tuning knob

For a target length `L`, some of the spectrum must stay sub-cycle: `2π · base^((D_h−2)/D_h) > L`, i.e.

```
base  ≳  (L / 2π)^(D_h / (D_h − 2))
```

Llama 3.1 ships `rope_theta = 500,000` for its 128K context — slowest cycle ≈ 2.6M tokens, leaving ~23% of channels monotone across the full window. Raising the base (or NTK/YaRN-rescaling an already-trained ladder) isn't an optimization; it's what makes the slow-channel escape hatch *exist*. Two related concessions in the wild: **partial RoPE** leaves a fraction of head dims unrotated — `θ = 0`, exactly position-invariant channels, at the cost of positional resolution (GPT-NeoX used `rotary_pct = 0.25`) — and some long-context designs drop positional encoding entirely in designated global-attention layers.

## Self-check

1. RoPE injects *absolute* position (each vector rotated by an angle tied to its own index) yet attention sees *relative* position. Walk through the two algebraic facts about rotations that make this happen.
2. Why is RoPE applied to `q` and `k` but not to `v`? And why does it not need to be added to the residual stream the way sinusoidal encodings are?
3. RoPE gives each coordinate pair its own frequency `θ_d = base^{−2d/D_h}` rather than rotating the whole vector by one angle. What breaks if you use a single frequency, and which pairs are responsible for RoPE's eventual extrapolation failure?
4. RoPE has zero learnable parameters, yet the model still controls how position-sensitive each of its features is. How — and what is the one thing it *can't* change?
5. Can a head escape a fast channel by choosing `q`/`k` directions "orthogonal to the rotation"? Why or why not?
6. At base 10000 and a 1M-token context, why can't training teach a head to attend to the first tokens (system prompt) from arbitrary positions — even though every rotation is deterministic?

### Answers

1. Two facts, both used in `(R(iθ)q)·(R(jθ)k) = q^T R(iθ)^T R(jθ) k`. **(a) A rotation's transpose is its inverse rotation:** `R(iθ)^T = R(−iθ)` (rotations are orthogonal, `R^T R = I`), so the query's absolute angle flips sign. **(b) Rotations compose by adding angles:** `R(−iθ)R(jθ) = R((j−i)θ)`. The two absolute angles combine into `(j−i)θ` — the offset — and the individual `i` and `j` vanish. Injecting absolute, reading relative, purely from these two properties.
2. `q` and `k` produce the attention *scores* (via `q·k`), which is the only place a positional relationship needs to be expressed; rotating `v` would just corrupt the retrieved content (the payload) with a positional phase that serves no purpose. It needn't touch the residual stream because RoPE is applied fresh inside every attention layer to the projected `q`/`k` — so position is recomputed where it's used, never has to survive the stack, and never competes with content for the `D` residual channels (contrast sinusoidal/learned absolute, which inject once at the input and hope the signal persists).
3. A single frequency is one clock hand: it wraps with a fixed period, so positions `i` and `i + period` get identical rotation and become indistinguishable — you lose the ability to resolve position across the whole range. The per-pair schedule fixes this by combining fast hands (fine local resolution) and slow hands (coarse global resolution) into a unique phase signature per offset. The **high-frequency pairs** (small `d`, large `θ_d`) are responsible for extrapolation failure: at positions past training they spin into phase values never seen during training, degrading attention — while the slow low-frequency pairs stay within their trained range.
4. Because RoPE is applied **after `W_Q` and `W_K`**, those learned projections determine each feature's coordinates and therefore **which frequency pair it lands in**. A feature routed to a slow pair is barely rotated even at long range (position-insensitive, good for semantics); one routed to a fast pair turns sharply with small offsets (position-sensitive, good for local/syntactic relations). What the model *can't* change is the menu itself — `θ_d` is a fixed schedule, so it can select among the available clock speeds (or blend across several pairs) but cannot invent a rate that isn't on the ladder. Note also that only *phase* is ever at stake: a rotation cannot change `‖q‖` or `‖k‖`.
5. **No.** Inside a rotating 2-D plane there is no fixed direction — a rotation moves every nonzero vector in the plane — and RoPE rotates all `D_h/2` planes at once, so no direction of head space escapes. The high-dimensional near-orthogonality intuition doesn't apply because the planes are fixed and axis-aligned; the rotation acts plane by plane, inescapably. The only control is **amplitude**: `W_Q`/`W_K` decide how much energy `‖q_d‖‖k_d‖` lands in each channel, and a channel at ~zero amplitude contributes nothing to the score. Gain control over a fixed filter bank — not direction-dodging.
6. Because the offset to those tokens **changes with where the query sits**. At `Δ ≈ 1M` every channel has wrapped (the slowest ~18 times at base 10000), so the 64 phases at any given `Δ` are pseudo-random — and a query at 700k versus 900k sees a *different* pseudo-random configuration for the same key. Making the channel sum large for **all** those `Δ` simultaneously is impossible at nonzero amplitude; deterministic phase is only exploitable at *fixed* offsets. So direct content matching to the prefix is structurally unavailable — what remains is attention-sink mass (position-blind, content-blind) and multi-hop relay through intermediate positions. The actual fix is a bigger base, so that some channels stay sub-cycle over the whole window.

## Exercise

Implement RoPE on a single head with `D_h = 8` (so 4 coordinate pairs), base 10000. Build the frequency vector `θ_d = 10000^{−2d/D_h}`, the `cos`/`sin` tables for positions `0..31`, and the rotate-in-pairs operation. Take two fixed random vectors `q`, `k`; place `q` at position `i` and `k` at position `j` and compute the RoPE'd dot product for several `(i, j)` pairs that share the *same offset* `i − j` (e.g. `(2,0)`, `(5,3)`, `(20,18)`). Confirm the score is (to floating point) identical across them — the relative property, empirically. Then hold `j = 0` and sweep `i` well past 31; watch the score for the *fast* pair start behaving erratically (unseen phase) while the *slow* pair stays smooth — a preview of the extrapolation failure that motivates [file 05](05_context_length_extension.md).
