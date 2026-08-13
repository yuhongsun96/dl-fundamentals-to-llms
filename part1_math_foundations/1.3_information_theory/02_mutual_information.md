# Mutual Information

## Definition

For random variables `X, Y`:
```
I(X; Y) = KL( p(x, y) ‖ p(x) p(y) )
        = H(X) - H(X | Y)
        = H(Y) - H(Y | X)
        = H(X) + H(Y) - H(X, Y)
```

**Intuition**: how much does knowing `Y` reduce your uncertainty about `X`? (And by symmetry, vice versa.) Zero iff `X ⊥ Y`; bounded above by `min(H(X), H(Y))`.

Units: nats or bits, same as entropy.

## Reading the four lines

The definition stacks four expressions that all equal the same number. Each is a different lens on "shared information."

| Form | Reads as |
|---|---|
| `KL( p(x,y) ‖ p(x)p(y) )` | How far the *true joint* is from the *pretend-independent* world `p(x)p(y)`. If `X` and `Y` were independent the joint would factor into the product, and this KL would be 0. So MI is literally "how much the joint deviates from independence." |
| `H(X) − H(X\|Y)` | Start uncertain about `X` (`H(X)`). After you observe `Y`, your leftover uncertainty is `H(X\|Y)`. The drop is what `Y` told you about `X`. |
| `H(Y) − H(Y\|X)` | Same thing read the other way — what `X` tells you about `Y`. That this equals the line above is *why MI is symmetric*. |
| `H(X) + H(Y) − H(X,Y)` | Inclusion–exclusion. Add the two uncertainties, subtract the joint so the overlap isn't double-counted. What's left is the overlap itself. |

**Why they're all equal** — one identity does all the work: `H(X,Y) = H(Y) + H(X|Y)` (the chain rule for entropy: total uncertainty = uncertainty in `Y` plus leftover uncertainty in `X` once `Y` is known). Rearranging gives `H(X|Y) = H(X,Y) − H(Y)`, and substituting into `H(X) − H(X|Y)` yields line 4. Swap the roles of `X` and `Y` and you get line 3. The KL form (line 1) is line 4 written out as `E[log p(x,y) − log p(x) − log p(y)]`.

**The Venn picture.** Draw two overlapping circles, areas `H(X)` and `H(Y)`. The union is `H(X,Y)`; the intersection is `I(X;Y)`; the crescent-shaped leftovers are the conditionals `H(X|Y)` and `H(Y|X)`. Every line above is one way of measuring the intersection from the circle areas.

**The two bounds, from the picture:**
- **`I ≥ 0`**: it's a KL, and KL is always ≥ 0 (the intersection can't have negative area). Zero exactly when the joint factors, i.e. `X ⊥ Y` — no overlap.
- **`I ≤ min(H(X), H(Y))`**: from `I = H(X) − H(X|Y)` and `H(X|Y) ≥ 0`, you get `I ≤ H(X)`; symmetrically `I ≤ H(Y)`. The intersection can't be bigger than either circle. The cap is hit when one variable is a deterministic function of the other (`H(X|Y) = 0` → knowing `Y` pins down `X` completely).

A concrete anchor: if `Y` is a noisy copy of `X`, `I(X;Y)` is large (most of `X`'s uncertainty is resolved by `Y`). If `Y` is a coin flip unrelated to `X`, `I = 0` (the joint is just the product). Everything in between is partial information.

## Why MI matters in representation learning

A good representation `z` of input `x` should be **informative about some target `y`**. Many self-supervised and contrastive methods are derivable as "maximize `I(z; y)`" under various approximations.

`I(X; Y)` is hard to compute exactly (requires knowing `p(x, y)`), so practical methods optimize variational lower bounds.

## InfoNCE — the practical MI lower bound

Given `N` samples where `(x_i, y_i)` are "positive" pairs (from `p(x, y)`) and `(x_i, y_j)` for `j ≠ i` are "negatives" (marginal product):

```
L_InfoNCE = -E[ log( exp(f(x_i, y_i)) / Σ_j exp(f(x_i, y_j)) ) ]
```

This is a softmax over the `N` candidates, with correct match as label. Minimizing `L_InfoNCE` maximizes a lower bound on `I(X; Y)`, with tightness growing in `N`.

### The intuition, step by step

**The problem it solves.** We want to maximize `I(X; Y)`, but MI needs the true joint `p(x, y)`, which we never have. InfoNCE sidesteps this with a trick that needs only a batch of data and a learnable score.

**The setup — one right answer in a lineup.** Take a batch of `N` pairs. The correctly-matched ones, `(x_i, y_i)`, are the **positives**. Deliberately mismatched ones, `(x_i, y_j)` for `j ≠ i`, are the **negatives** — and because you grabbed an `x` and an unrelated `y`, they're samples from the marginal product `p(x)p(y)`. So each `x_i` now sits next to its one true partner hidden among `N−1` impostors.

**The score.** Define `f(x, y)` = how compatible `x` and `y` are. In CLIP this is just the dot product of the two embeddings over a temperature: `f(x, y) = <image_emb, text_emb> / τ`. Higher = "these belong together."

**Read the loss as a question.** That fraction inside the `log` is a **softmax**: the numerator is the score of the *correct* pair, the denominator sums scores over *all* `N` candidates. So it answers "out of all candidates, what probability did the model put on the right one?" The `−log(...)` is exactly cross-entropy for the classification task **"which `y` matches this `x`?"**, with the true partner as the label.

**What minimizing it does.** To shrink the loss, the model must give the true pair a high score and the impostors low scores — i.e. pull matched embeddings to point the same way (high dot product) and push mismatched ones apart. No labels needed beyond "these two co-occurred."

**Why this bounds MI.** The only way to reliably win this pick-the-match game is for the embeddings to genuinely encode the information `X` and `Y` share. A model that does well provably captures a **lower bound on `I(X; Y)`**, so driving the loss down pushes that bound up. And more impostors make the game harder: with 2 candidates winning proves little; with 32K candidates it demands the embeddings carry a lot of shared information — which is why the bound tightens as `N` grows (see the next section).

**This is exactly CLIP's loss.** `f` is `<text_emb, image_emb> / τ`; positives are matched text-image pairs in a batch; negatives are other pairs in the same batch. CLIP builds the full `N × N` similarity matrix and runs this softmax to reward the diagonal (true pairs) and punish the off-diagonal (mismatches). So CLIP is, word for word, maximizing MI between the text and image modalities.

The same structure appears everywhere — only *what counts as a positive pair* changes; the score → softmax-over-batch → cross-entropy-toward-the-match machinery is identical:
- **SimCLR / MoCo** (vision SSL) — positives are two augmented views of the *same image* ("match the two crops"). MoCo's momentum queue exists to cheaply supply more negatives.
- **Sentence-transformer training** — positives are two sentences that should mean the same thing.
- **DPR and dense retrieval** — `X` is a query, `Y` is the passage that answers it ("find the right document").
- **E5, BGE, and other text embedding models** — the same query–document contrastive recipe at scale.

## "Large batch" for contrastive learning

InfoNCE's bound tightens as the number of negatives grows. That's why CLIP used 32K-batch, why SimCLR wanted big batches, and why MoCo introduced a momentum queue (effective batch of 65K). It's not an optimization artifact — it's information-theoretic.

## MI and the Information Bottleneck

Tishby's Information Bottleneck:
- Encoder should produce `z` that **maximizes `I(z; y)`** (informative about target).
- And **minimizes `I(z; x)`** (compresses away irrelevant info).

There's debate over whether real DL actually does this ("IB phase transitions"), but the framing is useful: a good representation is a lossy compression of the input that preserves what matters.

## Data Processing Inequality (DPI)

For any Markov chain `X → Y → Z`:
```
I(X; Z) ≤ I(X; Y)
```

"Post-processing can only destroy information, never create it." Consequence: any representation `z = f(x)` has `I(z; y) ≤ I(x; y)`. Your features can't contain more task-relevant info than the raw input.

This is why:
- Frozen pretrained encoders limit downstream performance — they've discarded some info.
- Distillation never exceeds teacher performance on the exact task.
- "Just predict pixels directly" is always an option at sufficient scale.

## Sufficiency vs. minimality

A representation `z` is **sufficient** if `I(z; y) = I(x; y)` (captures all task-relevant info). It's **minimal** if it contains nothing else. The ideal is both — hard to achieve simultaneously.

Most pretrained encoders are nowhere near minimal; they retain lots of info useful for many possible downstream tasks. That's the whole point of "general-purpose representations".

## MI estimators beyond InfoNCE

Worth being aware of:
- **MINE** (Mutual Information Neural Estimator)
- **NWJ** / **Donsker-Varadhan** bounds
- **Barber-Agakov** bound (= InfoNCE)

You won't derive these, but if you see them in a paper, they're all variational bounds on MI with different tightness/bias tradeoffs.

## Self-check

1. Express `I(X; Y)` in terms of entropies. Why is this symmetric in `X, Y`?
2. In CLIP, what are `X` and `Y`? Why does the per-batch softmax loss correspond to an MI lower bound?
3. The DPI says processing can only destroy information. So how does a neural net seemingly "create features" useful for classification?

### Answers

1. `I(X; Y) = H(X) + H(Y) - H(X, Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)`. **Symmetric**: swap `X ↔ Y` and the formula is unchanged. Intuition: "how much knowing one variable reduces uncertainty about the other" doesn't depend on which one you start from. The information shared between `X` and `Y` is a property of the joint distribution, not of an asymmetric "causal" relationship.
2. **`X` = text embedding, `Y` = image embedding** (or vice versa). The InfoNCE loss `-log(exp(sim(x_i, y_i)) / Σ_j exp(sim(x_i, y_j)))` does a softmax over `N` candidates with the matched `(x_i, y_i)` pair as the "correct" label. This is a **variational lower bound** on `I(X; Y)`: as the batch size `N` grows, the bound tightens. So minimizing the InfoNCE loss = maximizing a lower bound on the mutual information between text and image modalities. CLIP's training is literally "maximize I(text; image)" via this approximation. Same structure for SimCLR (image-image), DPR (query-document), sentence-transformers, etc.
3. The neural net doesn't *create* Shannon information — DPI is correct. What it does is **rearrange existing information into a more useful (e.g., linearly separable) form**. The label `y` is determined by the input `x`, so `I(features; y) ≤ I(x; y)`. But "useful" ≠ "Shannon-informative." A linear classifier might fail on raw pixels (high `I` but bad geometry) and succeed on learned features (lower `I`, but good geometry — task-relevant info aligned with linear directions). The net trades off: discards label-irrelevant info, preserves and reorganizes label-relevant info. Feature learning is about *organization*, not creation.

## Exercise

Implement a tiny InfoNCE loss for 2D data: sample `(x, y)` pairs with some correlation structure (e.g., `y = x + noise`), train small encoders for `x` and `y`, and minimize InfoNCE. Vary batch size from 16 to 1024 and watch downstream correlation of learned embeddings. This gives you hands-on intuition for why batch size matters for contrastive learning.
