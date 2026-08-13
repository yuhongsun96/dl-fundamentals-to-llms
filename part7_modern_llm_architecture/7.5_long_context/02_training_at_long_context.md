# Training at Long Context

[File 01](01_where_quadratic_bites.md) identified the training-side wall: a single long sequence's activations exceed one device, and data parallelism can't help because the batch is one sample. This file is how that's solved — the sequence dimension becomes a parallelism axis — plus the staged recipe that makes long-context training affordable.

**Scope:** the systems and scheduling side. The *positional* side — how RoPE frequencies are rescaled so a model trained short can run long — is fully worked in [5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md) (Position Interpolation, NTK-aware, YaRN) and assumed here.

## Why the sequence must be split

The standard parallelism axes each fail for a single long sequence:

| Axis | Splits | Why it doesn't help |
|---|---|---|
| Data parallel | the batch | The batch is **one** sequence |
| Tensor parallel | within-layer matrices | Helps weights/activations per layer, but every device still holds the full `S`-length activation for its shard |
| Pipeline parallel | layers | Each stage still processes the whole sequence |

So you split along `S`: device `i` owns tokens `[i·S/P, (i+1)·S/P)`. FFN and normalization are position-wise and split trivially ([5.2/01](../../part5_transformer_rebuilt/5.2_the_full_block/01_assembling_the_block.md)). **Attention does not** — every query needs every key, so the operation spans all shards. That single fact is the whole problem, and it has a beautiful solution.

## Ring attention

The key realization: [FlashAttention's online softmax](../7.2_efficient_attention/02_flashattention.md) already computes attention by streaming K/V tiles past a fixed Q block, carrying `(m, l, acc)` and renormalizing lazily. **Those tiles don't have to come from local memory.** They can arrive over the network.

Ring attention (Liu et al., 2023) arranges devices in a ring:

```
each device holds:  its Q shard (fixed), and one K/V shard (rotating)
repeat P times:
    compute partial attention of local Q against the currently-held K/V shard
    update the running (m, l, acc) via the online-softmax step
    pass the K/V shard to the next device; receive the next one
after P steps: every Q has seen every K/V → out = acc / l
```

Three properties make this work:

- **Exact.** It's the same online-softmax recurrence, so the result equals full attention to float precision — the exactness verified in [7.2/02](../7.2_efficient_attention/02_flashattention.md) carries over directly.
- **Overlappable.** Each device computes on the block it holds while the next block is in flight, so communication hides behind computation — provided compute per block exceeds transfer time, which is why block sizes are tuned to network bandwidth.
- **Memory per device is `O(S/P)`.** Context length scales with device count: the advertised route to 1M-token training.

This is the payoff of online softmax being *associative and incremental* — a property introduced for HBM traffic that turns out to be exactly what distributed attention needs. Worth noting as a general pattern: an algorithm restructured to stream from a slower memory tier usually also streams across a network.

**Causal load imbalance** is the practical wrinkle: with causal masking, the device holding the *last* tokens has far more unmasked work than the one holding the first (query at position `t` attends to `t` keys). Naive sharding leaves early devices idle. Fixes assign each device a *pair* of chunks — one early, one late — so per-device work is balanced (this is what "striped" or zigzag ring attention does).

You'll also see **Ulysses-style sequence parallelism** (DeepSpeed), which instead all-to-alls to switch from sequence-sharding to *head*-sharding for the attention operation and back — trading a different communication pattern for not needing a custom ring kernel. Both are in production; ring composes better with very long sequences, Ulysses with many heads.

## The staged recipe

Nobody trains 15T tokens at 128K context — it would be enormously wasteful, since `O(S²)` prefill compute and reduced token throughput apply to every token, and most training text is short anyway ([6.4/02](../../part6_pretraining_paradigms/6.4_training_mechanics_at_scale/02_packing_boundaries_loss_masking.md)). The universal recipe is staged:

1. **Bulk pretraining at short context** (4K–8K) for the overwhelming majority of tokens.
2. **A long-context extension stage** near the end: raise `S` (often in steps — 8K → 32K → 128K), rescale RoPE frequencies ([5.3/05](../../part5_transformer_rebuilt/5.3_positional_information/05_context_length_extension.md)), and train on a mixture deliberately enriched with genuinely long documents. Llama 3 used ~800B tokens across six such stages; the fraction of total budget is single-digit percent.
3. **Verify with retrieval-style evals**, not perplexity ([file 03](03_evaluating_long_context.md)).

This is the mid-training stage from [6.2/04](../../part6_pretraining_paradigms/6.2_data/04_mixtures_and_midtraining.md), and its data requirement is the underrated constraint: **you need documents that actually contain long-range dependencies.** Concatenating short documents to fill 128K teaches nothing about long range — and if cross-document attention is unmasked, it teaches the *opposite* lesson, that distant context is irrelevant noise ([6.4/02](../../part6_pretraining_paradigms/6.4_training_mechanics_at_scale/02_packing_boundaries_loss_masking.md)). Books, long code repositories, legal and scientific documents are scarce relative to the token budget, which is why synthetic long-context data ([6.2/05](../../part6_pretraining_paradigms/6.2_data/05_synthetic_data.md)) — inserted facts, cross-document reasoning chains — is a live area.

The two decisions from [6.4/02](../../part6_pretraining_paradigms/6.4_training_mechanics_at_scale/02_packing_boundaries_loss_masking.md) that matter most here, restated because this is the stage where they bind: **intra-document attention masking** and **per-document position resets**, adopted together.

## Why it matters in modern LLM work

- **It's why "context length" is a training-budget decision**, not a config flag: a model's real long-context ability reflects how many tokens it saw at length, on documents that had long-range structure.
- **Ring attention is the enabling technology** for the 1M-token claims, and knowing it's *exact* tells you those aren't approximations.
- **The online-softmax-generalizes lesson** is worth carrying: the same recurrence serves SRAM tiling, distributed attention, and streaming inference.

## Self-check

1. Why do all three standard parallelism axes fail for a single 1M-token sequence?
2. Which parts of a Transformer block shard trivially along the sequence, and which doesn't — and why?
3. Explain ring attention in terms of the online-softmax state, and state why it's exact.
4. What is causal load imbalance in ring attention, and what's the standard fix?
5. Why not just train the whole run at 128K? Give two reasons, one compute and one data.
6. A team fills 128K windows by concatenating short web documents and reports no long-context improvement. Diagnose.

### Answers

1. **Data parallel** splits the batch, but a single sequence is one sample — nothing to split. **Tensor parallel** splits within-layer weight matrices, but each device still materializes activations for the full `S` positions of its shard, so per-device activation memory still scales with `S`. **Pipeline parallel** splits layers, but each stage processes the entire sequence through its layers. None of them touches the dimension that's too big.
2. **FFN and normalization shard trivially** — both are position-wise, so a device can process its token range with no information from others ([5.2/01](../../part5_transformer_rebuilt/5.2_the_full_block/01_assembling_the_block.md)). **Attention doesn't**, because it mixes across positions: every query must be scored against every key, so a device holding only some keys cannot complete any query locally.
3. Each device keeps its Q shard fixed and its running `(m, l, acc)`; K/V shards rotate around the ring, and on each arrival the device performs one online-softmax update against the block it currently holds. After `P` rotations every Q has seen every K/V. It's exact because it is *literally the same recurrence* as tiled FlashAttention — the tiles just arrive over the network rather than from HBM — and that recurrence was verified to match full attention to ~1e-15 for any tile size ([7.2/02](../7.2_efficient_attention/02_flashattention.md)).
4. With causal masking, query position `t` attends to `t` keys, so the device holding late tokens has far more unmasked work than the device holding early ones — early devices finish and idle at every ring step, and the collective waits for the slowest. Fix: give each device a **pair of chunks, one early and one late** ("striped"/zigzag sharding), so total unmasked work per device is equalized.
5. **Compute:** attention is 82% of block FLOPs at 128K versus 6.5% at 2K ([file 01](01_where_quadratic_bites.md)), so every token costs far more and total throughput collapses — you'd buy a small fraction of the tokens for the same budget. **Data:** most text is short, so you'd be padding or concatenating rather than training on genuine long-range structure, spending the extra compute on documents that don't exercise the capability.
6. They trained on 128K windows containing no long-range dependencies — concatenated short documents have none — so there was nothing for the model to learn. Worse, if cross-document attention was unmasked, the training signal actively taught that distant context is irrelevant ([6.4/02](../../part6_pretraining_paradigms/6.4_training_mechanics_at_scale/02_packing_boundaries_loss_masking.md)). Fix: source genuinely long documents (books, repos) or synthesize long-range structure, and enable intra-document masking with position resets.

## Exercise

Implement ring attention on one machine to prove the exactness claim. (a) Simulate `P = 4` devices: split `K`/`V` into 4 shards and, for a fixed Q block, loop over shards updating `(m, l, acc)` with the online-softmax step from [7.2/02](../7.2_efficient_attention/02_flashattention.md). Verify the result matches full attention to ~1e-15, and confirm that **shard visitation order doesn't change the result** — the associativity that makes the ring valid. (b) Add causal masking and count unmasked score computations per shard-pair; tabulate per-device work for naive contiguous sharding versus paired early/late sharding, and report the imbalance ratio you removed. (c) Compute per-device activation memory for `S ∈ {128K, 1M}` at `P ∈ {1, 8, 64}` and identify the minimum `P` that fits 80 GiB. (d) One sentence: what network bandwidth would you need for communication to hide behind compute at your chosen block size?
