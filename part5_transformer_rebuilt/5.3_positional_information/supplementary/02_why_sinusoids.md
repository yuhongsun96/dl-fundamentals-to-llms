# Supplementary: Why Sinusoids, and How Far the Binary Analogy Goes

Companion to [`../02_sinusoidal_and_learned_absolute.md`](../02_sinusoidal_and_learned_absolute.md). The primary file states what sinusoidal `PE` is and gives the clock-hands picture. This file is the deeper layer: **where the design comes from**, how much of the popular "it's like binary counting" intuition is actually correct, and the two algebraic properties that make the scheme worth caring about.

**Convention:** row-vector, as in the primary file. `ω_d = 1/10000^(2d/D)` is the frequency of pair `d`; dims `2d` and `2d+1` hold `sin(i·ω_d)` and `cos(i·ω_d)`. Numbers below use `D = 64` unless stated, and are all reproducible in a few lines of torch.

---

## 1. Why simpler schemes don't work

The requirements are modest: a function `i ↦ PE(i) ∈ R^D` that is **bounded** (it's added to an embedding with O(1) entries), **unique per position**, and **defined for any `i`** with no table to run off the end of. Almost every simple idea fails one of them.

| Candidate | Why it fails |
|---|---|
| Add the raw index `i` to every dim | Unbounded. At `i = 4000` the position term is ~4000× the token signal — the embedding is erased. |
| Normalize: `i / max_S ∈ [0,1]` | Bounded, but resolution collapses — adjacent tokens differ by `1/8192`. Worse, a vector's *meaning* now depends on `max_S`, so it isn't portable across sequence lengths. |
| One-hot the slot | Needs `D ≥ max_S`, consumes the entire width, and encodes no notion of nearness: position 5 is as far from 6 as from 5000. |
| A single sinusoid `sin(i)` | Bounded and smooth, but **aliases** — `i` and `i + 2π` are identical. One clock cannot tell time past one revolution. |

That last failure is the productive one. A single clock aliases, so use **many clocks at different speeds**: fast ones to separate neighbors, slow ones to separate distant regions. That is the frequency ladder, and it is the same idea as place-value notation — which is where the binary analogy comes from.

---

## 2. The binary analogy: exactly how far it goes

The analogy is worth taking seriously because it is **literally true in a limiting case**, and then instructive about where it fails.

### The exact relationship

Binary counting *is* a sinusoidal ladder that has been thresholded. For bit `k` of position `i`:

```
bit_k(i) = step( sin( π(i + 0.5) / 2^k + π ) )
```

This holds exactly — verified for all `i < 32`, `k < 5`. Bit `k` toggles with period `2^(k+1)`, which is precisely a **square wave** at frequency `π/2^k`. So:

> **Binary place-value = a sinusoidal ladder with (a) power-of-two frequencies and (b) every value squashed to `{0,1}` by a threshold.** Sinusoidal `PE` is the same construction with a geometric frequency ratio instead of 2, and with the analog value kept instead of thresholded.

### What the analogy gets right

- **Multi-scale place value is the core idea.** Several "digits," each resolving a different scale; low digits distinguish neighbors, high digits distinguish regions.
- **Logarithmic width.** You need about `log(max_S)` digits, not `max_S` slots — 12 bits covers 4096 positions. This is why a modest `D` pins down thousands of positions, and why the one-hot scheme above is so wasteful by comparison.
- **Uniqueness comes from the combination.** No single dimension identifies a position; the full pattern across dimensions does.

If your question is *"why more than one frequency?"*, the binary intuition answers it correctly and you can stop there.

### Where it breaks

The moment you ask what the dimensions *do*, it misleads in four ways — the first being the one that matters.

**(a) The values are continuous, and that is the entire point.** There is no toggling and no thresholding. At the fastest frequency, `sin(i)` for `i = 0…7` runs `0, .841, .909, .141, −.757, −.959, −.279, .657` — a smooth traversal, not a bit flip. Smoothness means **nearby positions get nearby vectors**, so a model that learns to handle position 40 has learned something about position 41.

Binary is catastrophically bad at exactly this. Compare adjacent positions under both codes:

| pair | sinusoid cos-sim | binary Hamming distance |
|---|---|---|
| 0 vs 1 | 0.9662 | 1 bit |
| 7 vs 8 | 0.9662 | **4 bits** |
| 15 vs 16 | 0.9662 | **5 bits** |
| 127 vs 128 | 0.9662 | **8 bits** |

Every adjacent pair has *identical* similarity under sinusoids — 0.9662, regardless of where in the sequence you look (this is property 3.2 below). Under binary, "adjacent" costs anywhere from 1 to 8 bit flips depending on how many carries the increment triggers. Position 127 and 128 are neighbors that share **not a single bit**.

It gets worse — binary distance can't even order near vs. far:

| pair | true distance | sinusoid cos-sim | binary Hamming |
|---|---|---|---|
| 20 vs 25 | 5 | 0.7345 | 3 bits |
| 20 vs 70 | 50 | 0.4898 | 3 bits |

Ten times farther apart, **same** Hamming distance. The sinusoid correctly reports the more distant pair as less similar. A binary code is a set of unique labels whose geometry has nothing to do with position; the sinusoidal code is a geometric embedding of the number line.

**(b) The frequencies aren't powers of two.** The ratio between consecutive pairs is `10000^(2/D)` — at `D = 64` that's `1.33×` per step, not `2×`. Nothing wraps at a clean boundary, and there's no carry.

**(c) A pair is one angle, not two digits.** Dims `2d` and `2d+1` are not independent — they're constrained to the unit circle (`sin² + cos² = 1`), so the pair has *one* degree of freedom expressed in two coordinates. Binary digits are independent; you can set them arbitrarily. This is why "each dim is a bit" undercounts what's going on.

**(d) The payoff property has no binary analogue.** This is the deepest break. Sinusoids weren't chosen for uniqueness — a binary code is unique, and so is a learned lookup table. They were chosen because a **shift in position is a rotation** (§3.2, §4). There is no fixed linear operator that maps the bits of `i` to the bits of `i+1`; binary increment is carry propagation, which is nonlinear and position-dependent. The property that makes sinusoids interesting — and that leads directly to RoPE — is exactly the one the analogy cannot express.

### Verdict

**Keep it for "why multiple scales at different rates." Drop it as soon as you ask what a dimension holds or why the offset structure matters.** The better mental image is the one in the primary file: **analog clock hands**. Continuous, smooth, and rotation — the operation that turns out to matter — is native to them.

---

## 3. The two properties you get for free

Both are easy to derive and easy to check numerically, and together they're most of the reason the scheme works.

### 3.1 Every position has the same norm

Each pair contributes `sin²(iω_d) + cos²(iω_d) = 1`, so summing over `D/2` pairs:

```
‖PE(i)‖ = √(D/2)        for every i
```

With `D = 64` that's `5.657` at positions 0, 1, 100, and 511 alike — no drift at all. So **no position is louder than another**, and adding `PE` never destabilizes the residual stream. Contrast the raw-index scheme, whose injected magnitude grows without bound and eventually swamps the token content.

(Related practical note: the original paper multiplies the token embedding by `√d_model` before adding `PE`. With `‖PE(i)‖ = √(D/2)` fixed, that scaling is what keeps token content from being drowned out by a positional signal of comparable size.)

### 3.2 Dot products depend only on the offset

This is the important one. Using `cos(A − B) = cos A cos B + sin A sin B` on each pair:

```
PE(i) · PE(j) = Σ_d [ sin(ω_d i)·sin(ω_d j) + cos(ω_d i)·cos(ω_d j) ]
              = Σ_d cos( ω_d (i − j) )
```

The absolute positions `i` and `j` **cancel completely**. Whatever `i` is, the similarity between `PE(i)` and `PE(i+k)` is the same function of `k`. Numerically, `D = 64`, `k = 5`:

| `i` | `PE(i) · PE(i+5)` |
|---|---|
| 0 | 23.504 |
| 20 | 23.504 |
| 100 | 23.504 |
| 300 | 23.504 |

Identical to every digit, and matching the closed form `Σ_d cos(5 ω_d)`. The encoding is *absolute* — each position gets its own tag — but its **geometry is purely relative**. That gap between "absolute tag" and "relative geometry" is the whole subject of files 03–04: the relative structure is *present* but merely *available*, and RoPE is what makes attention obliged to use it.

---

## 4. Why each frequency needs both a sin and a cos

Three independent reasons, any one of which would be sufficient:

1. **A rotation needs a plane.** Shifting `i → i+k` rotates each pair by `ω_d k`, and rotation is a 2×2 operation. With sines only, `sin(ω(i+k))` cannot be written as a fixed multiple of `sin(ωi)` — you need the `cos(ωi)` component to complete the linear combination. Drop cos and the offset-only structure of §3.2 disappears, along with the entire path to RoPE.
2. **Sin alone is ambiguous.** `sin(θ) = sin(π − θ)`, so one sine value maps to two positions per cycle, and there's no way to distinguish an offset of `+k` from `−k`. The cos partner resolves the quadrant.
3. **Constant norm requires the pair.** `sin²` alone doesn't sum to 1 per pair, so §3.1 fails and different positions get different magnitudes.

Two dims spent on one frequency buy strictly more than one dim each on two frequencies.

---

## 5. The frequency ladder: spacing and the base 10000

The denominator `10000^(2d/D)` grows geometrically in `d` from `1` to ~`10000`, so `ω_d` shrinks from `1` to `1/10000`.

- **The direction is arbitrary.** Reversing `d` merely permutes dimensions, and the following operation is a learned linear map, which is permutation-agnostic. Don't look for meaning in "frequency decreases with `d`."
- **Geometric rather than linear spacing is the real design choice.** It covers four orders of magnitude in `D/2` steps. Linearly-spaced frequencies would cluster every clock at a similar speed: highly redundant dimensions, and no long-range coverage at all.
- **`10000` sets the slowest clock.** Max wavelength `2π · 10000 ≈ 62,832` tokens — deliberately far longer than any sequence the original model saw (512–2048). The slowest clock must **not complete a revolution** inside the trained length, or the coarse "where am I" signal would wrap and become ambiguous. The base is a context-length budget, which is precisely the knob that gets renegotiated to extend context ([file 05](../05_context_length_extension.md)).

Full ladder at `D = 64`:

| `d` | `10000^(2d/D)` | `ω_d` | wavelength |
|---|---|---|---|
| 0 | 1.00 | 1.0 | **6.3 tokens** |
| 1 | 1.33 | 0.750 | 8.4 tokens |
| 2 | 1.78 | 0.562 | 11.2 tokens |
| 8 | 10.0 | 0.100 | 62.8 tokens |
| 16 | 100 | 0.010 | 628 tokens |
| 24 | 1000 | 0.001 | 6,283 tokens |
| 31 | 7499 | 0.000133 | **47,117 tokens** |

---

## 6. What the slow end actually looks like

At the top of the ladder the angle is tiny: `ω ≈ 10⁻⁴`, so across 512 positions it never exceeds `0.068` radians. Small-angle approximations take over:

```
sin(x) ≈ x − x³/6 ≈ x        the sin dim becomes a near-perfect LINEAR RAMP in position
cos(x) ≈ 1 − x²/2 ≈ 1        the cos dim becomes a near-CONSTANT
```

Measured on the slowest pair (`d = 31`, `D = 64`):

| `i` | angle (rad) | `sin` | `cos` |
|---|---|---|---|
| 0 | 0.0 | 0.000000 | 1.00000000 |
| 1 | 0.000133 | 0.000133 | 0.99999999 |
| 100 | 0.013335 | 0.013335 | 0.99991109 |
| 511 | 0.068143 | 0.068090 | 0.99767917 |

Nothing mysterious happens at the slow end: the sin dim is **the raw position, linearly scaled down**, and its cos partner sits at ~1 everywhere (moving 0.0023 in total across the whole sequence). Monotone, no wraparound — the pure "early / middle / late" signal, at low resolution.

The fast end is the mirror image. At `d = 0`, wavelength 6.3 tokens: a large distinctive jump every step, and complete ambiguity beyond ~6 tokens.

**One honest consequence.** The *useful* dynamic range is narrower than `D` suggests. When `S` is far below the slowest wavelength, the top-of-ladder dims are nearly constant and carry little information — the ladder is partly spent on scales the sequence never reaches. That mismatch between the frequencies the formula provides and the frequencies a given context length needs is exactly the opening that the scaling methods in [file 05](../05_context_length_extension.md) exploit.

---

## Self-check

1. Binary counting is a thresholded sinusoidal ladder. Name the two things the thresholding destroys.
2. Positions 127 and 128 are adjacent but share no bits in common, while their sinusoidal cos-similarity is 0.9662 — the same as positions 0 and 1. Which property of §3 explains why *every* adjacent pair gives the same number?
3. Under a 12-bit binary code, positions 20↔25 and 20↔70 are both Hamming distance 3. Why is that disqualifying for a positional encoding, given that the code is still perfectly unique?
4. `PE(i) · PE(j)` depends only on `i − j`. Why is that not already sufficient to call the encoding "relative"?
5. Base 10000 gives a slowest wavelength of ~62,832 tokens. What specifically goes wrong if you train at `S = 100,000` without changing the base?
6. For the slowest pair, the cos dimension is ≈ 1 at every position in a 512-token sequence. If it carries almost no information there, why not drop it and spend the dimension on another frequency?

### Answers

1. **Smoothness and the rotation structure.** Thresholding to `{0,1}` makes the code discontinuous, so nearby positions no longer get nearby vectors and nothing generalizes from position `i` to `i+1`. It also destroys the algebra: a square wave has no `(sin, cos)` pair, so there is no rotation taking position `i` to `i+k`, and §3.2's offset-only dot product is gone. Power-of-two frequencies are a third difference, but a cosmetic one — a geometric ratio of 2 would still be a valid ladder.
2. §3.2: `PE(i) · PE(i+k) = Σ_d cos(ω_d k)`, which contains no `i`. The similarity of two positions is a function of their *offset* alone, so "adjacent" has one fixed similarity value everywhere in the sequence. Binary has no analogue because increment cost depends on carry propagation, which depends on the absolute value of `i`.
3. Because a positional encoding is not just an identifier — it's the input to a **dot product** inside attention. Attention scores are geometric, so what matters is whether the encoding's geometry reflects positional distance. A code that reports the same distance for offsets of 5 and 50 gives attention no usable signal about proximity, forcing the model to memorize the mapping from bit patterns to distances instead of reading it off. Uniqueness is necessary but nowhere near sufficient.
4. Because relative structure being *available* is not the same as it being *used*. The vector is still added to the input as an absolute tag, then has to survive 32 layers of residual-stream writes; nothing in the architecture requires attention to exploit the rotation, and the model may or may not learn to. Making the score depend on `i − j` *by construction* — rather than hoping the model discovers it — is what RoPE ([file 04](../04_rope.md)) achieves by moving the rotation into the `q·k` product itself.
5. The slowest clock completes more than one full revolution inside the trained length (100,000 > 62,832), so the coarse positional signal **wraps around**: positions roughly 62,832 apart receive nearly identical encodings across the entire ladder, and every clock is now ambiguous. Nothing catches the error — the model just cannot distinguish those positions. Raising the base (equivalently, stretching the wavelengths) is the fix, and it's the core of the NTK/YaRN-style scaling in [file 05](../05_context_length_extension.md).
6. Because its *value* isn't the point — its *pairing* is. Dropping it costs you both properties from §4 at that frequency: the shift-by-`k` rotation no longer closes, and `‖PE(i)‖` stops being constant. And "≈1 everywhere" describes *this* `S`, not the encoding — the formula is defined for any `i`, and near `i ≈ 15,000` that same cos dimension is swinging through its full range. The redundancy is deliberate insurance for sequences longer than the one in front of you. That said, the underlying observation is legitimate: a fixed geometric ladder does misallocate frequencies relative to any particular context length, which is exactly the slack §6 describes and [file 05](../05_context_length_extension.md) exploits.
