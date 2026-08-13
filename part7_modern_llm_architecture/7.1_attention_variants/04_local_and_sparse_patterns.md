# Local and Sparse Attention Patterns

The third attack on the KV cache ([file 01](01_the_kv_cache_problem.md)): don't compress what you store — **store less of it**, by not attending to everything. This is the oldest idea in efficient attention (2019–2020's "efficient Transformer" wave) and, after several years of being the *wrong* answer for LLMs, it came back in 2024–25 in a form that works. The arc is instructive: the same idea failed and then succeeded depending on one design decision.

## Sliding-window attention

Restrict each query to the most recent `w` tokens. Cache cost stops growing with `S` and pins at `w`:

```
full causal:     position t attends to [0, t]        cache = O(S)
sliding window:  position t attends to [t−w, t]      cache = O(w)
```

**Mistral 7B** (2023) shipped this at `w = 4096` in every layer, and its argument for why information still travels further than `w` is the important part: **receptive field compounds with depth.** Layer 1 sees `w` back; layer 2's inputs already summarize `w` back from each of those, so layer 2 effectively sees `2w`; after `L` layers the reachable span is `L · w` — for Mistral, `32 × 4096 = 131,072` tokens. This is exactly the CNN receptive-field argument, and exactly the multi-hop relay mechanism from [5.3/04](../../part5_transformer_rebuilt/5.3_positional_information/04_rope.md)'s failure analysis.

The catch — and it's the same catch as in that file: **reachable is not retrievable.** Information can propagate `L·w` tokens through `L` sequential hops, but each hop is lossy and bandwidth-limited (`d_model` values), so you cannot pull an exact 100k-token-old fact through 30 hops of summarization. Sliding windows are excellent for locality-dominated language modeling (perplexity barely moves) and poor for long-range *retrieval* — which is precisely the gap [7.5/03](../7.5_long_context/03_evaluating_long_context.md) shows perplexity fails to measure.

## Interleaving local and global layers

The fix that actually ships: **most layers local, some layers full.** A few global layers restore exact long-range access; the local majority keeps the cache small.

- **Gemma 2 / Gemma 3** alternate local (sliding-window) and global (full) attention layers — Gemma 3 pushes to a 5:1 local:global ratio with a 1024-token window, which is where most of its long-context memory savings come from.
- The pattern generalizes an older idea: **Longformer/BigBird** (2020) mixed local windows with a few "global tokens" attended by everyone.

The arithmetic is the point. With ratio `r` local layers per global layer, cache scales as `(r·w + S)/(r+1)` per layer instead of `S` — at `S = 128K`, `w = 1024`, `r = 5`: about **`(5×1024 + 131072)/6 ≈ 22.7K`** token-equivalents per layer instead of 131K, a ~5.8× cut, while every 6th layer retains exact global access. That's a much better quality/memory frontier than all-local, and it's why "hybrid local/global" beat "all sliding window" as the deployed design.

This also answers the practical question from Part 6's system-prompt discussion: with interleaved global layers, the prefix is always reachable *exactly* in some layer, in every block group — no reliance on relay.

## Attention sinks — the accident that had to be handled

A finding that started as a bug report and became standard practice. **StreamingLLM** (Xiao et al., 2023) observed that if you naively evict the oldest tokens to maintain a sliding window at *inference* on a model trained with full attention, quality collapses — and that keeping just the **first few tokens** (4 is typical) alongside the recent window restores it almost entirely.

The mechanism is softmax's normalization constraint. Softmax must distribute total weight 1 somewhere, even when a head has nothing it wants to attend to. Models learn to dump that surplus mass on a semantically-empty, universally-visible position — and position 0 is the canonical choice because every query can see it. Evict the sink and the surplus mass gets forced onto *real* tokens, distorting every head that was using the sink as a null option.

Two consequences worth carrying:

- **Practical:** streaming/windowed inference must retain sink tokens. This is now standard in serving stacks, and it's the same phenomenon as the "attention sinks help the system prompt stay attendable" observation — with the caveat that a sink is a *pressure valve*, not retrieval: the head isn't reading the prefix's content.
- **Architectural:** several models now add an explicit learned "sink" logit per head so the model has a designated null option and doesn't have to hijack token 0. Giving a mechanism its own parameter rather than letting it squat on a token is the cleaner design.

## The 2025 return of learned sparsity

The 2020 efficient-attention wave (Reformer, Longformer, BigBird, Performer…) largely lost: fixed hand-designed sparsity patterns cost quality, and FlashAttention ([7.2/02](../7.2_efficient_attention/02_flashattention.md)) then made *exact* attention fast enough that approximations weren't worth their quality risk. The 2024–25 revival differs on the design decision that matters — **the sparsity is learned and trained-in, not imposed post-hoc:**

- **DeepSeek NSA** (Natively Sparse Attention, 2025) combines compressed coarse tokens, selected fine-grained blocks, and a local window, with the selection **learned during pretraining** and hardware-aligned block sizes. Reported to match or beat full attention while being substantially faster on long sequences.
- The general lesson: *"train the model with the sparsity it will be served with."* Retrofitted sparsity fights a model that learned to rely on dense access; native sparsity lets the model allocate around it — the same principle as GQA needing uptraining ([file 02](02_mqa_and_gqa.md)) rather than being applied at inference.

Also in this family: **cross-layer KV sharing** (reuse one layer's K/V for several layers, attacking the `L` term in [file 01](01_the_kv_cache_problem.md)'s formula) — real savings, less adopted, and a reminder that every symbol in that formula has a paper attached to it.

## Why it matters in modern LLM work

- **`sliding_window` and layer-type fields in configs** are this file; reading them tells you a model's real long-context memory profile, which the advertised context length does not.
- **The reachable-vs-retrievable distinction** is the single most useful idea here, and it recurs in [7.3](../7.3_alternatives_to_attention/03_hybrids_and_recall.md) (SSM recall limits) and [7.5/03](../7.5_long_context/03_evaluating_long_context.md) (why NIAH exists).
- **The failure-then-success arc of sparsity** is a lesson in *when* to introduce a constraint: architectural changes generally must be present during training to be free.

## Self-check

1. Mistral's `L·w = 131,072` receptive field: state the mechanism, and the precise sense in which it overstates the model's long-range ability.
2. Compute the per-layer cache saving for a 5:1 local:global interleave with `w = 1024` at `S = 128K`. Why is this a better frontier than all-local?
3. What forces the existence of attention sinks, and why does evicting them hurt more than evicting an arbitrary middle token?
4. Fixed sparsity failed in 2020 but learned sparsity works in 2025. Name the design difference and the general principle.
5. A config lists `sliding_window: 4096` and 32 layers with no interleaving. What long-context behavior do you predict on (a) perplexity, (b) needle-in-a-haystack at 100K?

### Answers

1. Receptive field compounds with depth: each layer's inputs already summarize `w` tokens of context, so `L` layers reach `L·w`. It overstates ability because reach is achieved by **multi-hop summarization**, not direct access — each hop compresses into `d_model` values and is lossy, so exact retrieval of a specific distant fact (a name, a number) degrades sharply, even though the *influence* of distant tokens is nonzero. Reachable ≠ retrievable.
2. Per-layer average cache `= (5·1024 + 131072)/6 ≈ 22.7K` token-equivalents vs. 131K full — a **~5.8× reduction**. It beats all-local because the global layers preserve *exact* access to arbitrary distance, so retrieval-style tasks don't have to route through lossy hops; you pay full cache on only 1 layer in 6 and buy back the capability all-local sacrifices.
3. Softmax normalizes to 1, so a head with nothing worth attending to must still place its weight somewhere; models learn to park it on a universally-visible, low-content position (token 0). Evicting it removes the null option, so that mass is redistributed onto genuinely relevant tokens, corrupting the attention distribution of *every head using the sink* — a systematic, global distortion, unlike dropping one ordinary token which affects only heads that wanted it.
4. **Fixed/post-hoc vs. learned/native:** 2020 patterns were hand-designed and often applied to models trained (or expected to behave) densely; 2025 approaches learn *which* blocks to select during pretraining, so the model allocates its representations around the sparsity. Principle: architectural constraints should be present during training — a model trained dense has learned to depend on dense access, and removing it at inference is a distribution shift (cf. GQA needing uptraining).
5. (a) **Perplexity: nearly unaffected** — language is locality-dominated, and 4096 tokens of context captures most next-token predictability. (b) **NIAH at 100K: poor** — a fact 100K tokens back must survive ~24 lossy hops with no global layer to retrieve it exactly; expect sharp degradation with depth of burial. This gap is exactly why perplexity is a bad long-context metric ([7.5/03](../7.5_long_context/03_evaluating_long_context.md)).

## Exercise

Take the tiny GPT from the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb) and build a **synthetic retrieval task** that separates reachable from retrievable: a sequence containing `KEY <random 4-token id> ... [filler] ... QUERY KEY →` where the model must emit the id. (a) Train three variants at `S = 512`: full attention, sliding window `w = 64` in all layers, and a 3:1 local:global interleave. (b) Plot accuracy against the key-to-query distance. Full should be flat; all-local should fall off a cliff past roughly `L·w`; the interleave should track full closely. (c) Report each variant's KV bytes/token. (d) Then measure *perplexity* on ordinary text for all three and confirm it fails to distinguish them — you've reproduced, in miniature, why the field needed NIAH-style evals.
