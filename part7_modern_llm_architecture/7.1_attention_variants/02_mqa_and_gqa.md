# MQA and GQA — Sharing Keys and Values

The first and still most widely deployed answer to [file 01](01_the_kv_cache_problem.md): attack `n_kv` directly. The idea is almost embarrassingly simple, which is part of why it won — and the interesting content is *why* it costs so little quality, and why the industry converged on the specific number 8.

## The three configurations

Standard multi-head attention gives every query head its own key and value head: `H` queries, `H` keys, `H` values. The variants break that one-to-one:

| Scheme | KV heads | Cache size | Introduced |
|---|---|---|---|
| **MHA** | `n_kv = H` | baseline | Vaswani 2017 |
| **MQA** | `n_kv = 1` — *all* query heads share one K/V | `H×` smaller | Shazeer, 2019 |
| **GQA** | `n_kv = g`, groups of `H/g` queries share one K/V | `H/g ×` smaller | Ainslie et al., 2023 |

## How it works, shape by shape

The whole change is that `W_K` and `W_V` get **narrower outputs**, which creates a head-count mismatch that you then fix by broadcasting. Walk it with `H = 8` query heads, `n_kv = 2` KV heads, so the **group size** is `G = H/n_kv = 4` query heads per KV head:

**Step 1 — project, with asymmetric widths.**

```
W_Q : D → H·D_h        = 8·D_h    (unchanged — every query head keeps its own)
W_K : D → n_kv·D_h     = 2·D_h    ← narrower
W_V : D → n_kv·D_h     = 2·D_h    ← narrower
```

**Step 2 — the shape trace, in full.** Worth doing explicitly, because a common confusion is *where* the `n_kv` axis comes from. It is **not** produced by the matmul. Tracing `K` with `D = 32`, `n_kv = 2`, `D_h = 4` (so the projection's output width is `n_kv·D_h = 8`):

| Operation | Shape | What changed |
|---|---|---|
| input `x` | `(B, S, 32)` | — the trailing axis is the **feature** (a.k.a. hidden/channel/`d_model`) axis |
| `x @ W_K` | `(B, S, 8)` | **only the last axis**: `32 → n_kv·D_h = 8` |
| `.view(B, S, n_kv, D_h)` | `(B, S, 2, 4)` | ← **`n_kv` appears here**, splitting one axis into two |
| `.transpose(1, 2)` | `(B, 2, S, 4)` | head axis moves beside `B` |

So the matmul emits a single **flat** width of `n_kv·D_h`, and the `n_kv` axis is created by *reinterpreting* that width as two axes — the same free reshape (`view` splits the last axis right-to-left over a row-major buffer) worked through in the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb).

`Q` runs the identical pipeline, differing only in that its projection is `H·D_h` wide — so it emerges as `(B, H, S, D_h) = (B, 8, S, D_h)` against `K`/`V`'s `(B, 2, S, D_h)`. Attention requires one `K` head per `Q` head, and `8 ≠ 2`, so as written this doesn't typecheck. That mismatch *is* GQA.

**Step 3 — expand `K`/`V` by repeating each head `G` times.**

```
k_expanded = k.repeat_interleave(G, dim=1)   →  (B, 8, S, D_h)
```

which produces this assignment — verified `k_expanded[h] == k[h // G]` for every `h`:

| query head `h` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| uses KV head | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

So query heads 0–3 form one **group** sharing KV head 0, and 4–7 share KV head 1. That's the "grouped" in grouped-query attention.

> **Convention, and when it matters.** `repeat_interleave` gives kv indices `[0,0,1,1]` (contiguous groups); `repeat`/`tile` gives `[0,1,0,1]` (strided). These are **not** equivalent tensors — but they *are* equivalent models when training from scratch, because query heads have no intrinsic identity: the two groupings differ only by a relabeling of query heads, and permuting `W_Q`'s head slices together with `W_O`'s row blocks maps one to the other exactly (verified to `2.8e-14`). It's a gauge symmetry ([1.4/03](../../part1_math_foundations/1.4_optional_deeper_knowledge/03_bilinear_and_quadratic_forms.md)), so either convention trains to the same quality.
>
> Where it breaks is **weight compatibility.** A checkpoint's `W_Q` head `h` was trained to read KV head `⌊h/G⌋`; load it into code that expands with `repeat` and head `h` reads KV head `h % n_kv` instead — learned queries paired with the wrong keys, output cosine ≈ **0.36** against correct. HuggingFace/Llama checkpoints use `repeat_interleave`, so that's the convention to match when loading weights or when mean-pooling MHA heads during uptraining (below) — the pooling groups must be the same groups the expansion will pair.

**Step 4 — standard attention, unchanged.** `softmax(q kᵀ/√D_h) v` per head, same causal mask, same `W_O`. Head `h` computes `softmax(q_h · k_{⌊h/G⌋}ᵀ) v_{⌊h/G⌋}` — its **own** query against its **group's** shared key/value.

Two notes on what's really happening. **The expansion is conceptual, not physical**: a real kernel indexes into the `n_kv` heads rather than materializing an `H`-head copy, which is why the memory saving is real and not undone in step 3 — and why the *cache* stores 2 heads while the *math* sees 8. And **only the cache dimension shrinks**: you still compute `H` full attention distributions, so attention FLOPs are unchanged. GQA buys memory, not compute.

## Two immediate observations

**It shrinks parameters too**, since `W_K`/`W_V` are narrower — verified on a 70B-class config (`D = 8192`, `H = 64`, `D_h = 128`, `L = 80`):

| | attention params | KV cache/token |
|---|---|---|
| MHA (`n_kv = 64`) | 21.47B | 2,560 KiB |
| **GQA-8** | **12.08B** | **320 KiB** |
| MQA (`n_kv = 1`) | 10.91B | 40 KiB |

Note the asymmetry: parameters fall 1.8× while the cache falls **8×**. The parameter saving is a side effect; the cache is the motivation (and see self-check 5 for why the parameters barely matter).

**GQA interpolates.** `n_kv = H` gives `G = 1`, the expansion becomes a no-op, and you have exactly MHA; `n_kv = 1` gives `G = H` and you have MQA. Everything between is one dial. That's the whole contribution of the GQA paper — MQA was known and known to hurt; GQA found the knee.

## Why the quality cost is small

The natural objection: heads specialize ([5.1/04](../../part5_transformer_rebuilt/5.1_self_attention/04_multi_head_attention.md)), so forcing them to share keys should destroy that. Three reasons it doesn't, much:

- **Queries stay independent.** All `H` query heads keep their own `W_Q`, so each head still *asks its own question* — it just searches a shared index. Since attention's selectivity lives in `W_Q W_Kᵀ` as one bilinear form ([1.4/03](../../part1_math_foundations/1.4_optional_deeper_knowledge/03_bilinear_and_quadratic_forms.md)), and `W_Q` is per-head while `W_K` is shared, the *product* still differs per head. Sharing `K` constrains that product's rank structure but doesn't collapse it.
- **`W_O` stays per-head too.** Even with shared `V`, each head's output passes through its own slice of `W_O` ([5.1/01](../../part5_transformer_rebuilt/5.1_self_attention/01_qkv_projections.md)), so heads still write different directions into the residual stream.
- **Heads were redundant anyway.** The pruning literature ([5.1/04](../../part5_transformer_rebuilt/5.1_self_attention/04_multi_head_attention.md)) found many heads removable with little loss. GQA exploits the same slack more gracefully — degrade the KV resolution of all heads slightly, rather than deleting some entirely.

The empirical shape, from the GQA paper and every model since: **MQA (`n_kv=1`) measurably hurts** quality and destabilizes training; **GQA with a handful of groups is nearly free**. Hence the convention.

## Why 8

`n_kv = 8` appears on Llama-2-70B, Llama-3 (all sizes), Mistral, Qwen, and most of the field. It's overdetermined, which is the best kind of default:

1. **It's past the knee.** Going from `n_kv = 1 → 8` recovers essentially all of MHA's quality; going `8 → 64` buys little and costs 8× the cache.
2. **It matches tensor parallelism.** Serving typically shards across 8 GPUs. With `n_kv = 8`, each GPU owns exactly one KV head — no replication of KV across shards, no cross-device gather. With `n_kv = 1` (MQA), the single KV head must be *replicated* on all 8 devices, partly defeating the savings; with `n_kv = 4` on 8 GPUs, you replicate 2× or leave devices idle. **8 KV heads on 8-way TP is the aligned case.** This is a genuine systems constraint masquerading as a modeling choice.
3. **The savings are already most of what's available.** From [file 01](01_the_kv_cache_problem.md)'s 70B numbers: `n_kv=64 → 8` takes 2,560 KiB/token down to 320 — an 8× cut. Pushing to `n_kv=1` would gain another 8× (40 KiB) but that's a 320 GiB → 40 GiB → 5 GiB progression at 128K: the first step is the one that changes what's possible.

Note the ratio isn't fixed at 8 by accident of `H`: Llama-3-8B has `H=32` (4 queries per KV head), the 70B has `H=64` (8:1), the 405B has `H=128` (16:1). **`n_kv` is pinned at 8 and the sharing ratio grows with model size** — bigger models tolerate more sharing, and the TP alignment argument doesn't care about `H`.

## Converting an existing model: uptraining

You don't have to pretrain from scratch. The GQA paper's practical contribution: take an MHA checkpoint, **mean-pool** the key and value projections within each group to initialize the shared heads, then continue training for ~5% of the original compute. Quality lands close to MHA at GQA's cache size. This is why several 2023-era models shipped GQA variants of already-trained MHA models, and it's a good instance of the general pattern — architectural surgery plus brief continued pretraining is often far cheaper than a fresh run.

## Why it matters in modern LLM work

- **It's on nearly every model card you'll read**; `num_key_value_heads` in a HuggingFace config is this, and dividing `num_attention_heads` by it gives the sharing ratio.
- **It's the baseline that MLA and sparse attention must beat** ([file 03](03_multi_head_latent_attention.md), [file 04](04_local_and_sparse_patterns.md)) — "better than GQA at equal quality" is the standard claim to evaluate.
- **The TP-alignment argument generalizes:** architectural constants often encode a hardware or parallelism constraint. When a number looks arbitrary, ask what it divides evenly into.

## Self-check

1. Which projections change under GQA, and which stay per-head? Why does that preserve head specialization?
2. MQA is 8× smaller than GQA-8 and yet nobody ships it. Give the quality reason *and* the systems reason.
3. Llama-3's 8B, 70B, and 405B all use `n_kv = 8` but sharing ratios of 4:1, 8:1, and 16:1. What is being held constant, and why that rather than the ratio?
4. Explain uptraining and why mean-pooling is the natural initialization.
5. GQA cuts both cache *and* parameters. Why is the parameter saving nearly irrelevant to the motivation?

### Answers

1. Only `W_K` and `W_V` narrow (from `H·D_h` to `n_kv·D_h` output dims); `W_Q` and `W_O` stay full per-head. Specialization survives because each head keeps its own query projection — so the effective bilinear form `W_Q^h W_Kᵀ` still differs across heads even with `W_K` shared — and its own output slice of `W_O`, so it writes its own direction into the stream. What's lost is per-head *resolution of the index being searched*, not the per-head question or answer.
2. **Quality:** `n_kv = 1` forces all `H` heads through a single key/value subspace, which measurably degrades quality and was reported to destabilize training — the GQA paper's motivating finding. **Systems:** with 8-way tensor parallelism, one KV head must be replicated across all 8 devices (each needs the full K/V to compute its query shard), so you pay 8× the storage you nominally saved, or eat a cross-device communication. `n_kv = 8` on 8-way TP gives each device exactly one KV head — no replication, no gather.
3. **`n_kv = 8` is held constant**, so the sharing ratio `H/n_kv` grows with model size. It's the right invariant because the binding constraints attach to `n_kv`, not the ratio: cache bytes scale with `n_kv` ([file 01](01_the_kv_cache_problem.md)), and TP alignment wants `n_kv` = device count. That larger models tolerate ratios of 16:1 is the empirical bonus — more heads means more redundancy to spend.
4. Uptraining = convert an MHA checkpoint to GQA and continue pretraining briefly (~5% of original compute) instead of training fresh. Mean-pooling is natural because the shared head must serve all queries in its group, and the mean is the least-squares-optimal single vector for a group of vectors — it minimizes the summed squared deviation from what each query head *used* to be reading, giving the smallest initial perturbation to every head's behavior.
5. Because `W_K`/`W_V` are a small fraction of parameters — the FFN is ~75–80% of a block ([5.2/02](../../part5_transformer_rebuilt/5.2_the_full_block/02_the_ffn.md)) — so the saving is a rounding error on model size, and weights are a *fixed, amortized* cost anyway ([file 01](01_the_kv_cache_problem.md)). The motivation is the *per-sequence, per-token* KV cache, which is what limits batch size and therefore throughput.

## Exercise

Implement GQA from the MHA code in the [capstone notebook](../../numpy_pytorch/05_capstone_transformer_block.ipynb). (a) Add `n_kv` as a parameter: project `K`/`V` to `n_kv·D_h`, then expand with `repeat_interleave`, asserting that group `g` really feeds queries `g·H/n_kv … (g+1)·H/n_kv − 1`. Then settle the convention question empirically: build the `repeat` version too, confirm it gives a *different* output with the same weights, and then find the permutation `σ` of `W_Q`'s head slices (and `W_O`'s row blocks) that makes the two agree exactly — demonstrating they're the same model under relabeling. Finally, feed weights built for one convention to the other and measure the output cosine (~0.36), which is the real bug this distinction guards against. (b) Verify `n_kv = H` reproduces your MHA output exactly. (c) For `H = 32`, `D_h = 128`, `L = 32`, tabulate parameters and KV bytes/token for `n_kv ∈ {32, 8, 4, 1}`; confirm the cache scales linearly in `n_kv` while total parameters barely move — the quantitative version of self-check 5. (d) Train the tiny GPT at `n_kv ∈ {32, 8, 1}` on the same data and compare final loss: you should see 8 nearly matching 32, and 1 visibly worse.
