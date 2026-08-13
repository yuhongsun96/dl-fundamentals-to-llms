# Context-Length Extension

A model pretrained with RoPE at `S_train = 8192` ([file 04](04_rope.md)) will happily *run* on a 32,768-token input — the code doesn't crash, there's no table to run off the end. But the output quality collapses, because of the failures that closed the last file: at positions past training, the **high-frequency RoPE pairs** spin into phase values `pos · θ_d` the model **never saw** — and if the target length outruns the slowest cycle entirely, no sub-cycle channel remains for long-range matching at all. This file is about closing that gap *without* a full retrain — the family of tricks that stretch a trained RoPE model to a longer context by **manipulating its frequencies or its position indices** so that long test-time positions map back into phase regimes the model already understands, and so that a slice of the spectrum stays sub-cycle over the new window.

**Convention:** row-vector (`Y = X W`). RoPE acts per-head on `q, k ∈ R^(D_h)`, `D_h = 128` → 64 frequency pairs `θ_d = base^{−2d/D_h}` (see [file 04](04_rope.md)). 8B anchor `S_train = 8192` from [ARCHITECTURE.md](../../ARCHITECTURE.md); "extension factor" `s = S_target / S_train` (e.g. `s = 4` for 32K).

The mental model for the whole file: RoPE's positional signal is

```
angle  =  pos  ·  θ_d
           ↑        ↑
    which token   radians per step for coordinate pair d
    (0 … S−1)     (fixed ladder, θ_d = base^{−2d/D_h})
```

Two independent indices, worth keeping straight before reading on:

| Symbol | What it indexes | Range (8B config) | Varies along |
|---|---|---|---|
| `pos` | **which token** — its absolute slot in the sequence | `0 … S−1` (up to 8191) | the sequence |
| `d` | **which coordinate pair** inside that token's `q`/`k` vector | `0 … D_h/2−1` = `0 … 63` | the vector's dimensions |

`θ_d` is a **fixed constant per pair** (no learned parameters), measured in **radians per position step**, spanning a geometric ladder from `θ_0 = 1` down to `θ_63 ≈ 1.2 × 10⁻⁴`. So `angle` is "radians per step × number of steps" — the phase that pair `d` has accumulated by position `pos`:

| `d` | `θ_d` | Angle at `pos = 1000` |
|---|---|---|
| `0` (fastest) | `1.0` | 1000 rad — many full wraps |
| `32` (middle) | `0.01` | 10 rad — ~1.6 turns |
| `63` (slowest) | `~1.2 × 10⁻⁴` | 0.12 rad — barely moved |

Together they form an `(S, 64)` table of angles — precisely the `cos`/`sin` caches from [file 04](04_rope.md). Clock analogy: `θ_d` is how fast hand `d` ticks, `pos` is how many ticks have elapsed, `angle` is where that hand points for this token.

**Why this matters here:** you can shrink `angle` two ways — scale `pos` down, or scale `θ_d` down — and every method below differs only in *which* of those two knobs it turns and *how evenly* across the 64 frequencies.

## Position Interpolation (PI) — scale the positions (Chen et al., 2023)

The bluntest fix: **linearly squash the position indices** so the maximum test position lands back inside the trained range.

```
pos'  =  pos / s          # s = S_target / S_train
angle =  pos' · θ_d  =  (pos / s) · θ_d
```

Position 32,000 with `s = 4` is treated as position 8,000 — inside the trained range. Every angle the model ever computes now falls in a range it saw during pretraining, so no frequency wanders into unseen phase.

- **Cheap and simple** — one division. But it needs **a little fine-tuning** (a few hundred to a few thousand steps) to adapt, because you've changed the effective resolution the model was trained on.
- **The cost is fine detail.** Squashing positions by `s` also squashes the *spacing* between adjacent tokens by `s` — the fast pairs, which did the fine local resolution, now see neighboring tokens `s×` closer together in phase than they were trained on. You trade high-frequency (local) resolution for the ability to reach long range. Uniform scaling hits the frequencies that could least afford it.

## NTK-aware scaling — scale the base instead (spread it unevenly)

PI's flaw is that it treats every frequency the same. NTK-aware scaling ("neural tangent kernel"-motivated) fixes this by scaling the **base** rather than the positions:

```
base'  =  base · s^(D_h / (D_h − 2))            # base 10000 → a larger base
θ_d    =  base'^(−2d / D_h)                      # → each frequency shrinks, but unevenly
```

Because the frequencies are a geometric ladder in `d`, scaling the base **stretches the ladder non-uniformly**:

- **High-frequency pairs** (fast hands, fine local detail) are **barely touched** — their phase stays close to trained, so local resolution is preserved.
- **Low-frequency pairs** (slow hands, long-range position) are **stretched the most** — which is exactly what you want, since it's the long-range end that needs to reach further.

The interpolation "budget" is spent where it hurts least. This preserves local detail far better than PI and is often **training-free** (works zero-shot, or with minimal fine-tuning). The intuition: don't uniformly compress all positional information — compress the coarse end, protect the fine end.

## YaRN — refined per-frequency NTK + attention correction (Peng et al., 2023)

YaRN ("Yet another RoPE extensioN") is the current go-to at large multipliers. It refines the NTK idea in two ways:

1. **Per-frequency ("NTK-by-parts") interpolation.** Rather than one base-scaling formula, classify each frequency by whether its wavelength is short (fits many times inside the trained context — leave it alone), long (spans the whole context — interpolate it like PI), or in between (blend). This is a sharper version of "protect fast hands, stretch slow hands," applied band by band.
2. **A temperature / attention-scaling correction.** Stretching positions subtly changes the distribution of attention logits (the average magnitude shifts), which softmax is sensitive to. YaRN multiplies the logits by a small constant to restore the pre-extension entropy. This correction is cheap and consistently helps at large `s`.

The payoff is strong results at large multipliers (8×, 16× and beyond) with modest fine-tuning — extending an 8K model to 128K is a YaRN-shaped problem.

## The methods side by side

| Method | What it scales | How evenly | Needs fine-tune? | Main tradeoff |
|---|---|---|---|---|
| **Position Interpolation** | position index `pos` | uniformly (every freq the same) | yes (light) | sacrifices high-freq / local resolution |
| **NTK-aware** | the base (→ all `θ_d`) | unevenly (protects fast, stretches slow) | often no | slightly ad-hoc single formula |
| **YaRN** | per-frequency band + logit temperature | most refined (band-by-band + entropy fix) | light | more moving parts, but best at large `s` |

Read top to bottom as increasing sophistication in *how selectively* the interpolation budget is spent: PI spends it uniformly, NTK spends it unevenly by one formula, YaRN spends it per band and also patches the softmax side effect.

## Eval caveats — perplexity is not retrieval

A warning that matters more than any single method: **the standard extension eval (perplexity on long text) badly overstates how well these work.** Low perplexity at 32K means the model's *next-token predictions* are locally fluent across a long document — but fluency mostly needs *nearby* context, so perplexity can look great while the model is completely ignoring the far end of its window. The honest test is **long-range retrieval** — "needle in a haystack": plant a fact at position 3,000 and ask about it after 60,000 tokens. Extended models routinely pass the perplexity test and fail the needle test, especially in the middle of the context ("lost in the middle"). When you read "we extended to 128K," ask which eval — perplexity is cheap and flattering; retrieval is the one that reflects usable context.

This is the doorway to the long-context material in **Part 7.5**, which covers the systems side (ring / sequence parallelism to *fit* a 128K sequence's `O(S²)` attention across devices) and the harder question of *effective* context length versus advertised context length — the needle-in-a-haystack limits these RoPE tricks run into. And note the floor under all of it: **the tokenizer decides what "a token of context" even is** ([Part 5.4, tokenization](../5.4_tokenization/01_granularity_tradeoffs.md)) — a 128K-token window is a very different amount of *text* depending on how the tokenizer packs bytes into tokens, so context-length numbers are only comparable at a fixed tokenizer.

## Self-check

1. All three methods shrink `angle = pos · θ_d` at long positions. State the specific knob each turns, and why NTK's choice preserves local detail better than PI's.
2. Why does uniform Position Interpolation degrade *fine* (local) resolution rather than coarse (long-range) resolution?
3. A paper reports strong perplexity at 64K after YaRN extension. Why is this weak evidence that the model can actually *use* 64K of context, and what evaluation would you demand instead?

### Answers

1. **PI scales the position index `pos`** (uniformly: every frequency's angle divided by the same `s`). **NTK scales the base**, which because the frequencies form a geometric ladder in `d` shrinks them *unevenly* — high-frequency pairs barely change, low-frequency pairs stretch most. **YaRN scales per-frequency band plus a logit temperature.** NTK preserves local detail because the fast (high-frequency) pairs, which carry fine local resolution, are the ones it leaves nearly untouched — it spends the interpolation budget on the slow pairs that encode long range, which is exactly where extra reach is needed. PI, dividing every angle equally, compresses the fast pairs just as hard as the slow ones, blurring local structure.
2. PI compresses *all* positions by the factor `s`, so the *spacing* between adjacent tokens shrinks by `s` too. The fast (high-frequency) pairs are what distinguish neighbor from neighbor; after compression they see adjacent tokens `s×` closer together in phase than in training, so their ability to resolve *local* offsets is what erodes. The slow pairs encode coarse position and were changing so gently that squashing them barely matters — hence long-range coarse position survives while fine local detail is what you pay with.
3. Perplexity rewards locally-fluent next-token prediction, and fluency is dominated by *nearby* context — so a model can score well at 64K while effectively ignoring everything beyond the last few thousand tokens; the metric never checks whether distant information was actually used. Demand a **long-range retrieval / needle-in-a-haystack** eval: plant a specific fact deep in the context and query it from the end (and from the middle, where models are weakest). Passing that — not low perplexity — is what shows the extended context is *usable*, not just tolerated.

## Exercise

Take the `D_h = 8`, base-10000 RoPE from file 04's exercise and pick an extension factor `s = 4`. Implement all three angle schedules: (a) baseline `pos·θ_d`, (b) PI `(pos/s)·θ_d`, (c) NTK-aware with `base' = base · s^{D_h/(D_h−2)}`. For a fixed offset (say query at `pos`, key at `pos−1`) sweep `pos` from 0 out to `4·S_train` and plot, for the fastest pair and the slowest pair separately, the per-pair phase under each schedule. You should see: baseline's fast pair running off into unseen phase past `S_train`; PI pulling *both* pairs back but visibly compressing the fast pair's step-to-step change; NTK barely altering the fast pair while pulling the slow pair back. Write two sentences: which schedule best preserves the fast pair's trained behavior, and why that maps onto "preserves local resolution."
