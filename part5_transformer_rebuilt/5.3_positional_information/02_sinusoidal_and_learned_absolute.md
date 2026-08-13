# Sinusoidal and Learned Absolute Positions

The 2017 Transformer paper needed to pay the positional debt from [file 01](01_why_position_must_be_injected.md), and it offered two answers that dominated for four years: a **fixed sinusoidal** encoding and a **learned lookup table.** Both are *absolute* (they tag each slot `i` with an index-specific vector) and both are *additive at the input* (the vector is added to the token embedding before the stack). They are the baseline everything later is measured against — and understanding *why* they were superseded is the whole motivation for files 03–05.

**Convention:** row-vector (`Y = X W`); `X ∈ R^(B, S, D)`, so a positional encoding is a matrix `PE ∈ R^(S, D)` added row-wise. Dimensions from [NOTATION.md](../../NOTATION.md); 8B anchor `D = 4096`, max `S = 8192` from [ARCHITECTURE.md](../../ARCHITECTURE.md).

Both schemes plug in the same place:

```
x_i = embedding(token_i) + PE(i)         # PE(i) ∈ R^D, added to the token vector
                                         # x_i then enters the residual stream
```

The only difference between the two is how `PE(i)` is produced: a fixed formula, or a trained parameter.

## Sinusoidal (Vaswani et al., 2017)

Build `PE(i)` from sines and cosines of the position `i`, at a ladder of geometrically-spaced frequencies. For dimension pair `2d` / `2d+1` out of `D`:

```
PE(i, 2d)   = sin( i / 10000^(2d/D) )
PE(i, 2d+1) = cos( i / 10000^(2d/D) )        d = 0, 1, ..., D/2 − 1
```

Read the frequency term `10000^(2d/D)` as a **wavelength**: at `d = 0` the wavelength is `2π` (fast oscillation, changes every token), and it grows geometrically to `2π · 10000` at the top of the ladder (slow, barely moves across the whole sequence).

> Where the design comes from, why the binary analogy only half-works, and the algebra behind the two properties below: [supplementary/02_why_sinusoids.md](supplementary/02_why_sinusoids.md).

### Reading the formula: pairs, not alternating dimensions

The formula looks like "every dimension alternates sin, cos, sin, cos." The better reading is that **dimensions come in pairs, and each pair shares one frequency.** There are `D/2` frequencies, not `D`: dims `2d` and `2d+1` hold the sin and cos of the *same* angle `θ_d(i) = i · ω_d`.

And `(sin θ, cos θ)` is just **a point on the unit circle at angle `θ`**. So the whole object is:

> `PE(i)` is a bank of `D/2` clock hands. Hand `d` sits at angle `i · ω_d`. Position `i` is encoded as the simultaneous reading of all `D/2` clocks, each running at a different speed.

Each pair spends two coordinates because it takes two numbers to say where a hand points. (Layout footnote: the paper interleaves pairs as `[sin₀, cos₀, sin₁, cos₁, …]`; most implementations — and RoPE — put all sines first and all cosines second, pairing dim `d` with `d + D/2`. Same encoding up to a permutation of dimensions, which the next linear layer can't see.)

### The frequency ladder

One clock can't tell time past one revolution, so use many at different speeds:

- **High-frequency dims** (small `d`) resolve **fine** position — they distinguish token `i` from `i+1`, but wrap quickly, so they can't tell `i` from `i+50`.
- **Low-frequency dims** (large `d`) resolve **coarse** position — they barely change token-to-token but drift monotonically across the sequence, placing you in "early / middle / late." At the very slowest, the angle stays so small that `sin` is essentially a linear ramp in position and `cos` sits at ~1.

| `d` (at `D = 64`) | `ω_d` | wavelength |
|---|---|---|
| 0 | 1.0 | **6.3 tokens** |
| 8 | 0.1 | 62.8 tokens |
| 16 | 0.01 | 628 tokens |
| 24 | 0.001 | 6,283 tokens |
| 31 | 0.000133 | **47,117 tokens** |

Stack the ladder and every position gets a **unique fingerprint** with fine and coarse information both present — "roughly where" from the slow dims, "exactly where" from the fast ones. The **geometric** spacing is what covers four orders of magnitude in `D/2` steps, and the base `10000` is chosen so the slowest clock never completes a revolution inside the trained length (max wavelength ≈ 62,832 tokens) — no wraparound, no ambiguity.

### Doesn't adding position corrupt the token?

It should worry you that `PE(i)` is *added* to the token embedding: both occupy the same `D` coordinates, so the positional value overwrites whatever the embedding put in each dimension. Measured on a real embedding table with matched magnitudes, that's not an exaggeration — adding `PE` changes the **median coordinate by 137%**, and position exceeds content in **60%** of coordinates. And yet the correct token is still recovered from the sum by nearest-neighbour search over a 49,152-token vocabulary **100%** of the time.

Both are true because of one fact about high-dimensional space:

> **High dimensions let you perturb everything while barely moving in any particular direction.**

Nothing downstream ever reads a coordinate — `W_Q`, `W_K`, `W_V` and every FFN key compute a *projection* `x · w` across all `D` dimensions. A vector of norm 3.18 contributes only ~0.10 along any given direction (`≈ 1/√D` of its length), because it's spread across all of them. Meanwhile the token's own embedding contributes its full `‖e‖²` along its own read-out direction. Signal adds **coherently**, interference **incoherently** — a `√D` margin.

There's a second layer of protection, and it's why the *low-rank* structure of `PE` matters: the positional signal occupies only ~225 of 512 directions (14 for half its energy), and `W_Q`/`W_K`/`W_V` are **learned** — so they can preferentially orient away from that subspace when they want pure content, or into it when they want position. Low-rank isn't just cheap to store; it's easy to avoid.

Full treatment with measurements: [1.4/04](../../part1_math_foundations/1.4_optional_deeper_knowledge/04_linear_readout_and_identifiability.md).

### Two properties worth knowing

- **Every position has the same norm:** `‖PE(i)‖ = √(D/2)`, since each pair contributes `sin² + cos² = 1`. So no position is louder than another and adding `PE` never destabilizes the residual stream.
- **Dot products depend only on the offset:** `PE(i) · PE(j) = Σ_d cos(ω_d (i − j))`. The absolute positions cancel. With `D = 64`, `PE(i)·PE(i+5) = 23.504` at `i = 0, 20, 100, 300` — identical to every digit.

The second is the one that matters — it's what "The clever property that hinted at the future" below unpacks, and the reason each frequency needs *both* a sin and a cos.

### Is it a binary encoding?

Partly — and the failure of the analogy is instructive. It's exactly right that binary is a **thresholded** sinusoidal ladder: bit `k` of `i` is a square wave that toggles with period `2^(k+1)`, so binary place-value is this same construction with power-of-two frequencies and every value squashed to `{0,1}`.

So the analogy captures the important idea — **multiple scales, fast digits for neighbors and slow digits for regions, uniqueness from the combination rather than any single dim.** If that's your question, stop there.

But the values here are **continuous, and that is the whole point.** Nothing toggles. Smoothness means nearby positions get nearby vectors, and binary is catastrophically bad at exactly that:

| pair | sinusoid cos-sim | binary Hamming |
|---|---|---|
| 0 vs 1 | 0.9662 | 1 bit |
| 127 vs 128 | 0.9662 | **8 bits** — adjacent positions sharing no bits at all |
| 20 vs 25 (distance 5) | 0.7345 | 3 bits |
| 20 vs 70 (distance **50**) | 0.4898 | 3 bits — *same as distance 5* |

Every adjacent pair has identical sinusoidal similarity no matter where you look (that's the offset-only property above), while binary's cost of "adjacent" swings from 1 to 8 bits depending on carries — and Hamming distance can't even order near vs. far. A binary code is a set of unique *labels*; the sinusoidal code is a geometric *embedding of the number line*, which is what a dot product inside attention needs.

**So: keep the analogy for "why multiple scales," drop it the moment you ask what a dimension holds.** Clock hands are the better image — continuous, smooth, and rotation is native to them, which is the subject of the next section. Full treatment in [supplementary/02_why_sinusoids.md](supplementary/02_why_sinusoids.md).

### The clever property that hinted at the future

Sinusoids have a rotation identity: `sin(i + k)` and `cos(i + k)` are *linear combinations* of `sin(i)` and `cos(i)` with coefficients depending only on `k`:

```
[ sin(i+k) ]   [ cos(kω)   sin(kω) ] [ sin(i) ]
[ cos(i+k) ] = [ −sin(kω)  cos(kω) ] [ cos(i) ]     (per frequency ω)
```

That 2×2 matrix is a **rotation by angle `kω`** — a function of the *offset* `k` alone, not of the absolute position `i`. So `PE(i+k)` is a fixed linear map of `PE(i)`, and in principle a linear attention projection could exploit this to attend "`k` steps back" regardless of where it is. This is the first appearance of the idea that will win: **absolute encoding whose structure secretly carries relative information.** RoPE ([file 04](04_rope.md)) takes exactly this rotation and moves it from the input into the attention dot product, where the relative property becomes exact rather than merely available.

### Why it's OK but not great

Two structural weaknesses, both traceable to "added at the input":

1. **Position must survive the residual stream.** The signal is injected once, at layer 0, then has to persist — unmolested — through 32 layers of attention and FFN writes that are all doing other things to the [residual stream](../5.1_self_attention/01_qkv_projections.md). Position and content share the same `D` channels and compete for them. There's no mechanism forcing the positional signal to stay legible at layer 30; the model has to *learn* to preserve it.
2. **Mediocre extrapolation.** The formula is defined for any `i`, so it doesn't hard-fail past training length — but the model only ever *learned* to read positions in the trained range. Beyond it, the low-frequency dims drift into phase combinations the attention weights never saw, and quality degrades. Better than the hard wall below, but not a real answer to train-short-run-long.

## Learned absolute (BERT, GPT-2)

The simplest possible alternative: don't compute `PE(i)` — **learn it.** Keep a trainable table

```
P ∈ R^(max_S, D)        # one learned D-vector per absolute position
PE(i) = P[i]            # pure lookup, exactly like the token embedding table
```

and add `P[i]` to token `i`'s embedding. Gradients flow into `P` like any other parameter; the model discovers whatever positional geometry helps the task.

- **Simpler and marginally better in-distribution.** No frequency schedule to justify, and empirically a hair better than sinusoidal *within* the trained range because the model tunes each position vector to the data instead of accepting a fixed formula.
- **A hard context wall.** The table has exactly `max_S` rows. Position `max_S` (0-indexed past the end) has *no embedding* — there is no `P[max_S]` to look up. The model cannot process a sequence one token longer than it was trained on without adding a randomly-initialized, never-trained row. **Zero extrapolation, by construction.** BERT's famous 512-token limit is exactly this table's number of rows.

## The two, side by side

| | Sinusoidal | Learned absolute |
|---|---|---|
| `PE(i)` | fixed `sin`/`cos` formula | trainable lookup `P[i]` |
| Parameters | **0** | `max_S · D` (e.g. `8192 · 4096 ≈ 34M`) |
| Where injected | added at input | added at input |
| In-distribution quality | good | slightly better |
| Beyond training length | defined but degrades (soft fail) | **undefined — hard wall** |
| Relative-position structure | latent (the rotation identity) | none — every slot independent |
| Used by | original Transformer, some enc-dec | BERT, GPT-2, GPT-3 |

Both are **absolute and additive**, and that shared design is the root of their shared limitation.

## Why the field moved past both

Three complaints, in order of importance, set up the next two files:

1. **Language cares about *relative* position, not absolute slot.** The relationship between a verb and its subject depends on how far apart they are and in which direction — not on whether the phrase starts at token 4 or token 4004. Absolute encodings force the model to *learn* to convert absolute indices into the relative offsets it actually needs, and to relearn it at every position. [File 03](03_relative_positions.md) encodes the offset `i − j` directly.
2. **Extrapolation.** Neither trains-short-runs-long well — one degrades, one hard-fails. The LLM era made long context a headline feature, so this went from a footnote to a dealbreaker.
3. **Injecting at the input is the wrong place.** Position competes with content for residual-stream channels and must survive the whole stack. Putting it *inside attention* — where position actually gets used, on `q` and `k` right before the dot product — is cleaner. [File 03](03_relative_positions.md) biases the attention logits; [file 04](04_rope.md) rotates `q` and `k`. Neither touches the residual stream.

## Self-check

1. In the sinusoidal ladder, which dimensions carry "roughly where in the sequence" and which carry "exactly which token"? Tie your answer to wavelength.
2. A learned absolute table trained at `max_S = 512` is handed a 513-token input. What exactly goes wrong, and why is this a *harder* failure than sinusoidal's degradation?
3. The sinusoidal rotation identity means `PE(i+k)` is a fixed linear function of `PE(i)`. Why is that suggestive of relative encoding — and why isn't it *actually* relative encoding as used at the input?
4. Why does each frequency need **both** a sin and a cos dimension? What specifically breaks if you build `PE` from `D` sines at `D` different frequencies instead?

### Answers

1. **Coarse ("roughly where")** lives in the **low-frequency / long-wavelength** dimensions (large `d`): they drift slowly and monotonically across the sequence, so their value tells you early/middle/late but not the exact token. **Fine ("exactly which")** lives in the **high-frequency / short-wavelength** dimensions (small `d`): they change every token, pinning down `i` locally, but wrap around fast so they alias across long distances. The full ladder combines both into a unique per-position fingerprint.
2. There is no row 513 in the table. `P` has shape `(512, D)`, so position 512 (0-indexed, the 513th token) indexes past the end — you must either truncate the input or bolt on a fresh, randomly-initialized, never-trained vector, which the stack has never learned to read. Sinusoidal at least *defines* `PE(512)` via the formula and its neighbors were trained on similar phases, so it degrades gracefully; the learned table has literally no value there, so it's an out-of-bounds error, not a quality dip.
3. It's suggestive because the map from `PE(i)` to `PE(i+k)` is a rotation that depends only on the offset `k`, not on `i` — so the encoding *contains* a clean relative structure that a linear attention could latch onto to implement "look `k` back." It isn't actually relative in use because the encoding is still added to the input as an *absolute* tag and then must survive the residual stream; nothing in the architecture forces attention to exploit the rotation, so the relative structure is merely *available*, not *guaranteed*. RoPE fixes this by moving the rotation into the `q·k` dot product, where the offset-only dependence becomes an identity the score *must* obey.
4. Three things break, any one of which is fatal. **Rotation:** a shift `i → i+k` is a rotation, which is a 2×2 operation and needs a plane — with sines only, `sin(ω(i+k))` isn't a fixed multiple of `sin(ωi)`, so the offset-only structure (and the path to RoPE) disappears. **Ambiguity:** `sin(θ) = sin(π−θ)`, so a lone sine maps to two positions per cycle and can't distinguish `+k` from `−k`; the cos partner resolves the quadrant. **Norm:** `sin²` alone doesn't sum to 1 per pair, so positions stop having equal magnitude. Two dims on one frequency buy strictly more than one dim each on two frequencies. (Expanded in [supplementary §4](supplementary/02_why_sinusoids.md).)

## Exercise

Compute the sinusoidal `PE ∈ R^(128, 64)` matrix (positions 0–127, `D = 64`, base 10000) and plot it as a heatmap with position on one axis and dimension on the other. You should see fast stripes on the low-`d` side smoothly stretching to slow bands on the high-`d` side — the frequency ladder made visible.

Then confirm the two properties numerically:

1. **Constant norm** — `‖PE(i)‖` should equal `√(D/2) = 5.657` for every `i`, to float precision.
2. **Offset-only dot products** — `PE(i) · PE(i+k)` at fixed `k = 5` should be *identical* at `i = 0, 20, 100` (you should get `23.504`) and match the closed form `Σ_d cos(ω_d · k)`. Repeat with cosine similarity for `i = 20`, `k = 0..100` — the same statement, normalized.

Finally, articulate in one sentence why a *learned* table would show no such `k`-only structure unless the data happened to teach it.

([Supplementary §6](supplementary/02_why_sinusoids.md) has the worked numbers for the slow end of the ladder if you want to check the small-angle behavior too.)
