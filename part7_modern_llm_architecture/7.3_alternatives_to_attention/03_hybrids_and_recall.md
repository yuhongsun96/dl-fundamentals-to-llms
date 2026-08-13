# Hybrids and the Recall Tradeoff

The verdict file for 7.3. Linear attention and SSMs deliver everything they advertise — constant-memory inference, flat throughput at long context — and still have not replaced attention. The reason is one specific, well-measured capability gap, and the field's response was not to pick a winner but to **mix**, in ratios that turn out to be remarkably consistent across labs.

## The gap: in-context recall

The failure is not "long-range dependencies" in the vague sense; SSMs handle those fine. It's **associative recall** — reproducing a specific token or span from earlier in the context, exactly. The canonical probe:

```
... A→4 ... B→9 ... C→2 ...        [many distractors]        ... B→?
```

Softmax attention solves this structurally: the query for `B` matches the key at `B`'s position and copies its value. This is the **induction head** mechanism (Part 11.2), and it's the mechanistic basis of in-context learning — few-shot prompting, copying formats, using a name introduced 20K tokens ago.

A fixed-state model cannot do this in general, and the reason is the pigeonhole argument from [7.3/01](01_linear_attention.md): the state is a sum of outer products `Σ φ(kₜ)vₜᵀ`, so it superposes all associations into `O(D²)` numbers. Retrieval works while keys stay near-orthogonal and interferes once the number of stored pairs exceeds capacity. The **Zoology** and **Based** line of work (Arora et al., 2023–24) established this quantitatively: performance on associative recall tracks **state size**, gated-convolution/linear-attention models lag attention specifically on this skill, and much of the remaining perplexity gap between SSMs and Transformers is attributable to recall-heavy tokens. The tradeoff is real, measured, and *fundamental* — not an artifact of insufficient tuning.

Note the asymmetry that makes hybrids attractive: **attention is only needed for a minority of tokens.** Most next-token predictions are locality-dominated (an SSM handles them at `O(1)` cost); a few require exact retrieval. So you don't need attention everywhere — you need it *somewhere*.

## The hybrid answer

Interleave layers. A few full-attention layers supply exact retrieval; the SSM/linear majority carries the sequence cheaply. What's striking is the convergence on ratios:

| Model | Composition | Attention share |
|---|---|---|
| **Jamba** (AI21, 2024) | Mamba + attention + MoE, 1 attention layer per 8 | **~1:8** |
| **Zamba** (Zyphra, 2024) | Mamba backbone + a *shared* global attention block reused periodically | ~1 block, reused |
| **Samba** (Microsoft, 2024) | Mamba + sliding-window attention, alternating | ~1:2 with local attn |
| **Nemotron-H / MiniMax-01** (2024–25) | mostly linear/SSM layers with periodic full attention | ~1:7 to 1:8 |

Roughly **one attention layer in 6–8** is the recurring answer, and it produces most of the memory win: with 1 attention layer in 8, KV cache drops ~8× versus a full Transformer while exact retrieval remains available in every block group. That's the same arithmetic as local/global interleaving ([7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md)) — and the same *principle*: a capability you need occasionally should be provided by a minority of layers rather than by every layer or none.

Zamba's variant is worth noting as the extreme: **share one attention block's parameters** across its several uses, so you pay the attention capability once in parameters and several times in compute.

## Where each family wins

| Workload | Prefer |
|---|---|
| Very long context, throughput-critical, summarization/streaming | SSM / hybrid |
| Constant-memory or on-device inference | SSM / hybrid |
| Heavy in-context learning, few-shot, exact copying, long-context retrieval | attention (or hybrid with enough of it) |
| Frontier general-purpose quality | attention-dominant, still |
| Non-language modalities with long smooth signals (audio, genomics) | SSM often wins outright |

The honest 2025 status: **no pure SSM is at the frontier of general language modeling**, and every serious SSM-family production model is a hybrid. Meanwhile hybrids are genuinely shipping — the question has shifted from "will attention be replaced?" to "what's the right attention fraction?", which is a much better question and one with an empirical answer per workload.

## The synthesis: this is Part 4's table, resolved

Return to [4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)'s tradeoff table. The RNN had cheap inference and bad training and recall; the Transformer had expensive inference and great training and recall. The 7.3 family fixed the RNN's *training* problem (linearity ⇒ parallelism, [file 01](01_linear_attention.md)) and its *forgetting* problem (input-dependent gates, [file 02](02_state_space_models.md)) — but the **fixed-state recall ceiling was never a training artifact.** It's the pigeonhole principle, and no algorithm removes it.

So the resolution is architectural pluralism: use a compressed state for the 90% of the job that's compression, and keep exact access for the 10% that's retrieval. Which is, read at the right altitude, what a Transformer's own division of labor already looks like — attention retrieves from the sequence, the FFN retrieves from fixed weights ([5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md)). Hybrids just add a third tier: a compressed running summary.

## Why it matters in modern LLM work

- **It's the framework for evaluating any "attention is dead" claim**: ask what the state size is and what the associative-recall numbers are. Those two questions predict most of the result.
- **Hybrid ratios are a real hyperparameter** you'll see in model cards, and the ~1:7 convergence is a useful prior.
- **It's the clearest case study in the curriculum of a *fundamental* rather than *contingent* limitation** — most architectural weaknesses get engineered away; this one is a counting argument.

## Self-check

1. State the recall failure precisely, and why "SSMs can't do long-range dependencies" is the wrong description of it.
2. Give the counting argument for why a fixed-state model cannot guarantee exact retrieval, and name the earlier file where the same argument appears.
3. Why does one attention layer in eight recover most of the capability, and what's the memory win?
4. Which two of the RNN's three historical defects did 7.3 fix, and which is unfixable?
5. Zamba shares one attention block across several uses. What is it economizing, and what is it not?
6. A team reports a pure-SSM model matching a Transformer on perplexity. What do you ask for before believing the models are equivalent?

### Answers

1. The failure is **associative recall** — retrieving a specific value bound to a specific key seen earlier, exactly (the induction-head/copying capability). "Long-range dependencies" is wrong because SSMs propagate long-range *influence* perfectly well; what they can't do is *address* an arbitrary earlier token and read it out precisely. Aggregate/smooth long-range signals are fine; pointwise retrieval is not.
2. The state is `Σₜ φ(kₜ)vₜᵀ`, a sum of outer products living in `O(D²)` numbers — a fixed budget independent of `S`. Storing `n` distinguishable key-value associations requires enough near-orthogonal directions; once `n` exceeds that capacity, retrievals interfere by the pigeonhole principle. Same argument as [4.1/03](../../part4_sequence_modeling_lineage/4.1_rnns_and_lstms/03_why_they_lost.md)'s fixed-state bottleneck, and it's the low-rank superposition limit of [1.4/01](../../part1_math_foundations/1.4_optional_deeper_knowledge/01_dimension_span_and_rank.md).
3. Because exact retrieval is needed for a *minority* of tokens, and one attention layer per block group provides it for all of them — information can be routed to the attention layer, retrieved exactly, and carried onward. The memory win is direct: only 1 layer in 8 maintains a KV cache, so cache size falls ~8× versus a full Transformer while retaining exact access somewhere in every group.
4. Fixed: **sequential training** (removing the recurrence's nonlinearity restores parallelism via scan/convolution) and **indiscriminate forgetting** (input-dependent gates — Mamba's `Δₜ`, GLA's learned decay — restore selective retention). Unfixable: the **fixed-state recall ceiling**, because it's a counting bound on how many associations a bounded state can keep separable, not a limitation of the optimizer or the kernel.
5. It economizes **parameters** — one attention block's worth of `W_Q/W_K/W_V/W_O` serves several positions in the stack — while still paying the **compute** and the **KV cache** at each use. So it's a parameter-efficiency play, not a memory or FLOP one; it bets that the *function* attention performs is reusable across depths even if its inputs differ.
6. Ask for **associative-recall and long-context retrieval evals** (needle-in-a-haystack, RULER, few-shot in-context learning), plus the **state size**. Perplexity averages over mostly-local tokens and systematically under-weights the recall-heavy minority where the families diverge — the same "perplexity is not retrieval" caveat as [7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md) and [7.5/03](../7.5_long_context/03_evaluating_long_context.md). Equal perplexity with unequal recall is the expected result, not a surprise.

## Exercise

Measure the tradeoff and then price the hybrid. (a) Build the associative-recall task from [7.3/01](01_linear_attention.md)'s exercise (`n` key-value pairs, then query one) and evaluate three tiny models at matched parameter count: full attention, a selective-SSM/linear-attention model, and a hybrid with 1 attention layer in 4. Plot accuracy against `n`. Expect: attention flat, SSM collapsing near its state capacity, hybrid tracking attention. (b) For each, compute KV bytes/token and per-token inference FLOPs at `S = 32K` using [7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)'s formula — the hybrid should land close to the SSM on memory and close to attention on accuracy, which is the entire thesis. (c) Sweep the hybrid ratio (1:2, 1:4, 1:8, 1:16) and find *your* knee; compare to the field's ~1:7. (d) One paragraph: for a streaming on-device assistant versus a long-document RAG system, which ratio would you ship and why?
