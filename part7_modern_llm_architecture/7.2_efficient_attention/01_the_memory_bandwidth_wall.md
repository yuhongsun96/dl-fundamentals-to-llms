# The Memory-Bandwidth Wall

[7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md) established that decode is memory-bound. This file makes that quantitative and general, because it's the lens that explains why FlashAttention exists, why MoE is fast, and why half of "optimization" work in LLMs is about moving fewer bytes rather than doing less arithmetic. If you internalize one number from Part 7, make it the machine balance.

## Arithmetic intensity and the roofline

Every kernel has a ratio:

```
arithmetic intensity  =  FLOPs performed / bytes moved to-and-from memory
```

Every accelerator has a corresponding ratio, its **machine balance**:

```
machine balance = peak FLOP/s / peak memory bandwidth
```

For an H100 (~990 TFLOP/s bf16 dense, ~3.35 TB/s HBM) that's **≈ 296 FLOP/byte**. The **roofline** rule follows immediately: a kernel with intensity below the machine balance is **memory-bound** — it cannot reach peak FLOPs no matter how good the math is, because the memory system can't feed it. Above it, **compute-bound**.

This means "how many FLOPs does it do" is often the *wrong* first question. The right one is "how many bytes must it touch, and how many FLOPs does it extract per byte?"

## The three regimes in an LLM

Run the numbers for a 70B dense model and the picture is stark:

| Phase | FLOPs | Bytes moved | Intensity | Verdict |
|---|---|---|---|---|
| **Decode, batch 1** | `2N ≈ 1.4e11` | weights `≈ 1.4e11` | **≈ 1** | wildly memory-bound (~300× off) |
| **Decode, batch `B`** | `2NB` | weights `≈ 2N` (read once, reused) | `≈ B` | needs `B ≈ 300` to balance |
| **Prefill, `S` tokens** | `2NS` | weights `≈ 2N` | `≈ S` | compute-bound for `S ≳ 300` |

The single structural fact: **weights are read once per forward pass regardless of how many tokens go through it.** So intensity is essentially "tokens processed per weight-read." Prefill has `S` of them; decode has one (per sequence). That's the whole asymmetry, and it produces two rules:

- **Prefill: minimize FLOPs.** Sparsity, better kernels, and lower-precision matmuls all pay.
- **Decode: minimize bytes.** Quantization, fewer active parameters (MoE, [7.4](../7.4_mixture_of_experts/01_the_sparse_ffn.md)), smaller KV cache ([7.1](../7.1_attention_variants/01_the_kv_cache_problem.md)), and larger batches all pay — and FLOP reductions that *add* memory traffic can make things worse.

Batching is the bridge: it converts decode's problem into prefill's by reusing each weight read across `B` sequences. Which is why the KV cache is *the* serving constraint — it's what stops you from reaching `B ≈ 300`.

**The causal chain, stated once because it's easy to get backwards:**

```
smaller KV cache → more sequences fit in memory → larger batch → higher intensity → better utilization
```

Each link earns its place. **Memory enables batching** — a sequence costs `S × KV bytes/token` of HBM, so whatever memory the cache doesn't consume is what's available to admit another sequence. **Batching raises utilization** because the weights are read from HBM *once per forward pass regardless of how many sequences are in flight*: with `B` sequences, that single read of each weight now feeds `B` multiply-accumulates instead of one. Same bytes hauled in, `B` times the arithmetic extracted from them — which is precisely `intensity ≈ B`.

Note the direction: the KV cache does not *enable* batching, it **competes with it** for the same memory. Shrinking the cache (GQA/MLA, [7.1](../7.1_attention_variants/02_mqa_and_gqa.md)–[03](../7.1_attention_variants/03_multi_head_latent_attention.md); paging, [file 03](03_kv_cache_memory_management.md); quantization, Part 9.2) is how you buy the batch size that buys the utilization.

## Where attention specifically hurts

Attention has a second, independent memory problem beyond the KV cache, and it's the one FlashAttention solves. The naive implementation materializes the score matrix:

```
S_mat = Q Kᵀ        # (S, S) — written to HBM
P     = softmax(S_mat)   # read (S,S), write (S,S)
O     = P V         # read (S,S)
```

That `S × S` matrix per head is quadratic in *memory*, not just compute — and it gets written and re-read several times. Verified for `S = 8192`, `D_h = 128`: the score matrix is **256 MiB per head** (fp32), against just **6 MiB** for streaming `Q`, `K`, `V` once — a **~43× traffic ratio**, growing linearly with `S`.

This is why pre-2022 attention was slow *even at moderate `S` where its FLOPs were a minority* ([7.5/01](../7.5_long_context/01_where_quadratic_bites.md) shows attention is only ~22% of block FLOPs at `S = 8192`). It wasn't the arithmetic; it was the round trips. Which reframes the whole "efficient attention" literature: many 2020-era methods reduced attention's *FLOPs* while leaving its memory traffic intact, and were therefore slower than a well-written exact kernel — the fate [7.1/04](../7.1_attention_variants/04_local_and_sparse_patterns.md) described.

## The memory hierarchy that makes tiling work

Why can a kernel avoid those round trips at all? Because on-chip memory exists and is enormously faster:

| Level | Size (H100-class) | Bandwidth |
|---|---|---|
| Registers / SRAM (per SM) | ~228 KB shared memory per SM | ~tens of TB/s |
| HBM (device memory) | 80 GB | ~3.35 TB/s |
| Host RAM | ~TB | ~tens of GB/s (PCIe) |

The rule: **HBM is the enemy; SRAM is free by comparison.** Any computation you can restructure to keep its working set in SRAM avoids the wall. Attention's working set is `Q`, `K`, `V` tiles plus a running accumulator — small. The `S × S` scores are what don't fit, so the trick is to *never form them in full*. That's [file 02](02_flashattention.md).

## Why it matters in modern LLM work

- **It's the correct mental model for "is this optimization worth it?"** Compute FLOPs and bytes, take the ratio, compare to ~300. Most disappointing optimizations are FLOP reductions applied to memory-bound code.
- **It explains the whole shape of Part 7 and 9.** GQA, MLA, MoE, quantization, FlashAttention, continuous batching — every one is a bytes-reduction or a byte-reuse play.
- **The prefill/decode split is a real engineering boundary** — separate kernels, separate batching policies, sometimes separate hardware (Part 9.2), because the two phases sit on opposite sides of the roofline.

## Self-check

1. Define arithmetic intensity and machine balance, and state the roofline rule in one sentence.
2. Why is decode's intensity ≈ 1 regardless of model size? Show it from `2N` FLOPs and `2N` bytes.
3. A colleague proposes an attention variant that halves attention FLOPs but materializes an extra `(S, S)` intermediate. Predict the wall-clock effect at `S = 8192` and justify with numbers.
4. Why does increasing batch size help decode but do nothing for prefill's intensity?
5. FlashAttention performs *slightly more* FLOPs than a naive implementation (it recomputes things). Why is it several times faster?

### Answers

1. **Arithmetic intensity** = FLOPs ÷ bytes moved for a given kernel; **machine balance** = peak FLOP/s ÷ peak bandwidth for the hardware (~296 FLOP/byte on H100). Roofline rule: if intensity < machine balance the kernel is memory-bound and cannot approach peak FLOPs; if greater, it's compute-bound.
2. One token through a dense model costs `≈ 2N` FLOPs (one multiply-add per parameter, [6.3/01](../../part6_pretraining_paradigms/6.3_scaling_laws/01_power_laws_kaplan.md)) and requires reading every weight once — `2N` bytes in bf16. The ratio is `2N / 2N = 1`, and `N` cancels: **every dense model at batch 1 has intensity ≈ 1**, so bigger models are not relatively worse here, they're all equally starved.
3. Almost certainly **slower**. Attention FLOPs are only ~22% of block FLOPs at `S = 8192`, so halving them saves ~11% of arithmetic on code that isn't compute-bound; meanwhile an extra `(S,S)` intermediate adds ~256 MiB per head of HBM traffic against a ~6 MiB streaming baseline. You've cut the cheap resource and spent the scarce one.
4. Because batching amortizes the *weight read* across sequences: weights are read once per forward pass, so `B` sequences give `B` tokens per weight-read and intensity `≈ B`. Prefill already has `S` tokens per weight-read from a single sequence, so it's already high-intensity and compute-bound — batching adds throughput but doesn't change the ratio that determines which side of the roofline you're on.
5. Because it trades the abundant resource for the scarce one: extra FLOPs (recomputing the score tile in the backward pass) in exchange for never writing the `S × S` matrix to HBM. On memory-bound code, arithmetic is nearly free and traffic is everything — the ~43× traffic reduction at `S = 8192` dominates a small FLOP increase.

## Exercise

Measure the wall yourself. (a) Write two attention implementations in PyTorch: one that explicitly materializes `Q @ K.T`, softmaxes, and multiplies; one that calls `F.scaled_dot_product_attention`. Time both for `S ∈ {512, 2048, 8192}` at `D_h = 128`, and plot the speedup against `S` — the ratio should *grow* with `S`, which is the traffic-ratio prediction, not a FLOP prediction. (b) For each `S`, compute predicted bytes moved for both and check the ratio matches the timing trend. (c) Compute your own GPU's machine balance from its spec sheet and locate decode-at-batch-1, decode-at-batch-64, and prefill-at-2048 on the roofline. (d) One sentence: which of the three would you optimize first, and would you target FLOPs or bytes?
