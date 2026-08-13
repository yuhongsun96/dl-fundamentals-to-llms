# Gradient Checkpointing: Trading Compute for Memory

## The activation-memory problem

Training a Transformer takes far more memory than running inference on it. The reason is that backward needs the forward activations cached.

For a model with `L` layers, residual dim `D`, batch `B`, sequence `S`, the activation memory for *one* forward pass is roughly:
```
M_act ~ O(L · B · S · D · k)
```

`k` is "how many tensors per layer we have to save" — typically 10–20 for a Transformer block (attention scores, Q, K, V, FFN intermediates, etc.). For a 7B model with `L = 32`, `D = 4096`, `B = 1`, `S = 8192`, this is on the order of tens of GB — often dwarfing the parameter memory itself.

## Standard backprop's space-time tradeoff

Standard backprop saves every activation on the forward pass, then walks the graph backward using them. The tradeoff:
- **Time**: forward + backward = ~3× forward FLOPs (file `01`).
- **Memory**: O(activations along the entire network) = the big number above.

The dominant cost in fitting a large model on a given GPU is *not* the parameters or gradients — it's the activations.

## The checkpointing idea

What if we don't save the activations? Then backward can't compute the local Jacobians, because they need the activations.

The fix: **save a few activations (the "checkpoints"), and recompute the rest from those checkpoints when backward needs them.**

Concretely: pick a subset of layers as checkpoints. During forward, save activations only at those layers — discard everything in between. During backward, when we need activations for a non-checkpointed segment, **rerun the forward pass through that segment** to regenerate them, then use them for backward.

Trade space for time:
- **Memory**: O(checkpoints), not O(activations).
- **Time**: each non-checkpoint segment is forwarded twice — once in the original forward, once during backward. Total forward work goes up by roughly one extra forward pass over the recomputed segments.

## The classic √N checkpointing strategy

For a network of `L` layers, the optimal even spacing places `√L` checkpoints. Each checkpoint stores activations every `√L` layers. Then:
- Memory: `O(√L)` activations instead of `O(L)`.
- Recompute cost: each segment is recomputed once, total extra forward work is `O(L)` — i.e. one extra forward pass over the whole network.

Net training step cost (was `3F`, now `4F`) for a `√L` reduction in activation memory. Often a great deal.

### Why `√L` and not "every N layers for a factor-N win"?

The factor-N intuition counts only one of the two things that occupy memory. Checkpointing has **two** activation costs that live at the same time:

1. **The stored checkpoints.** If you checkpoint every `N` layers, you keep `L/N` of them alive for the whole backward pass.
2. **The recompute working set.** To get the activations for a segment during backward, you rerun forward through that segment — which transiently materializes all `N` layers' worth of activations inside it. You can't avoid this: backward through a segment needs every activation in it at once.

So peak activation memory is the **sum**, not just term 1:
```
M(N)  ~  L/N   +   N
         └┬─┘     └┬┘
      checkpoints  recompute buffer
```

Now the "factor-N" claim falls apart. Push `N` large to shrink the checkpoints (term 1), and the recompute buffer (term 2) grows to match. At the extreme `N = L` (one checkpoint, the input), term 1 is `O(1)` but term 2 is `O(L)` — you've just *moved* all the memory from storage into the recompute buffer and saved nothing.

Minimize the sum: `dM/dN = -L/N² + 1 = 0  ⟹  N = √L`, giving `M = √L + √L = O(√L)`. That's the floor for single-level checkpointing — and it's `√L`, not `L`, precisely because the recompute buffer term stops you from spending freely on coarser spacing. (You *can* beat `√L` with **nested/recursive** checkpointing — checkpoint within recomputed segments — reaching `O(log L)` memory at `O(L log L)` compute. The `√L` result is the optimum for the simple one-level scheme.)

Note the compute side is flat across all `N`: every layer is recomputed exactly once during backward regardless of spacing, so the extra work is `O(L)` (one forward pass) no matter where you put the checkpoints. `N` is purely a memory knob — which is why you choose the `N` that minimizes peak memory, i.e. `√L`.

### Why "every block" works anyway for Transformers

Per-block checkpointing uses `L/1 = L` checkpoints — far *more* than the `√L` the memory-optimal analysis prescribes. Doesn't that contradict the above? No, because that analysis assumed every layer's activations cost the same. In a Transformer they don't: the **block-boundary** activation is just the residual stream (`B·S·D`), while the activations *inside* the block — the `(B,H,S,S)` attention scores, the `4D` FFN intermediates, the per-op buffers — are many times larger (the `k ≈ 10–20` factor from the memory formula above). So checkpointing every block keeps term 1 small (each checkpoint is only the cheap boundary tensor) while freeing the expensive within-block activations. You're not minimizing `L/N + N` in the abstract — you're exploiting that boundary activations ≪ internal activations, which is a property of the architecture, not of the spacing math.

For Transformer training specifically, you usually checkpoint at the **block boundary** — save the input to each Transformer block and recompute everything inside (attention scores, FFN intermediates, residuals) during backward. This is what HuggingFace's `gradient_checkpointing_enable()` does by default. It's not exactly `√L` spacing — it's "every block" — but the savings are dramatic because most of the activation memory lives inside the block.

## Selective recomputation

A finer-grained version: rather than recomputing everything in a segment, only recompute the activations that are **cheap to recompute relative to their memory cost**.

Concrete example: attention scores `(B, H, S, S)` are big (quadratic in `S`) and cheap-ish to recompute (one matmul). Many implementations selectively recompute attention while keeping FFN intermediates cached. FlashAttention essentially does this automatically — its tiled algorithm never materializes the full `(S, S)` attention matrix in the first place, so there's nothing to checkpoint or recompute. See Part 7.2.

## When to use it

Always-on for large-scale training. The memory savings let you:
- Use larger batches (better throughput).
- Use longer sequences (longer context).
- Fit larger models on the same hardware.

The compute cost is a known fixed overhead (~33% extra forward work, give or take) and is dwarfed by the throughput gains.

For small models or single-GPU prototyping, you can skip it.

## How it interacts with other things

- **Mixed precision (file 05)**: orthogonal. Checkpointing reduces activation memory; mixed precision reduces the bytes-per-activation. Both compose.
- **ZeRO / FSDP (Part 12)**: orthogonal. Those shard parameters / gradients / optimizer states across GPUs; checkpointing reduces per-GPU activation memory.
- **Pipeline parallelism**: synergistic. Pipeline parallelism slices the model across devices; activations have to be saved at pipeline boundaries (the "bubble"). Checkpointing inside each pipeline stage further reduces the per-stage activation cost.

## The PyTorch API in one line

```
from torch.utils.checkpoint import checkpoint
out = checkpoint(my_function, *inputs, use_reentrant=False)
```

`my_function` runs without saving activations; on backward, it's rerun. `use_reentrant=False` is the modern, recommended mode.

For Transformers, HuggingFace wraps this:
```
model.gradient_checkpointing_enable()
```

Sets a flag that causes each Transformer block to be wrapped in `checkpoint` during forward.

## A subtle correctness pitfall

Recomputation must be **deterministic**. If the recomputed forward produces different activations than the original (e.g. because of dropout's RNG), the backward will use a different set of activations than the gradient was computed against — silently wrong gradients.

Fix: PyTorch's `checkpoint` saves and restores the RNG state for the wrapped function, so dropout fires identically on recompute. But: BatchNorm in training mode (which updates running statistics) and any module with side effects are still hazardous. Most modern Transformers don't use dropout heavily (file 2.5/01) and don't use BatchNorm at all, so this is a non-issue in practice — but worth knowing.

## Self-check

1. Standard training step costs `3F`. With per-block gradient checkpointing on a Transformer, what's the new cost? Why isn't it `4F`?
2. Why does checkpointing usually pair well with mixed-precision training, rather than competing with it?
3. If checkpointing recomputes activations during backward, why is the result still bitwise-equivalent to non-checkpointed backward?

### Answers

1. With per-block checkpointing, every Transformer block is forwarded twice: once in the original forward, once when its segment is recomputed during backward. The original forward was `F`, the backward is `2F`, and we add one extra forward pass through the blocks during backward = `+F`. Total: `4F` — exactly as predicted. The intuition "isn't it 4F" is correct here. The reason the more general √L analysis gives a smaller overhead in some accounts is that `√L` checkpointing doesn't recompute every block — only the segments between checkpoints, with shared cost — but for "checkpoint every block", you really do pay `+F` extra. The win is in memory, not FLOPs.
2. They reduce different costs. Mixed precision (file 05) halves the bytes per activation — from fp32 (4 bytes) to fp16/bf16 (2 bytes). Checkpointing reduces the number of cached activations. They multiply: if checkpointing gives 2× memory reduction and bf16 gives 2×, you get 4× combined. They don't conflict at all — the activations you do choose to cache are just stored in 2 bytes each. Modern large-scale training stacks always use both.
3. PyTorch's checkpoint saves the RNG state of all relevant generators (CPU, CUDA) before the wrapped function runs, and restores it before the recompute. Any randomness (dropout, stochastic depth) reproduces exactly. As long as the wrapped function is otherwise deterministic given its inputs and the RNG state — true for almost all standard layers — the recomputed activations are bitwise identical to the originally computed ones. So the gradient computation that uses them is identical to what non-checkpointed backward would have computed. The price is purely compute, not correctness.

## Exercise

Take a small Transformer (e.g. `nanoGPT`-style, 6 layers, `D=384`, `S=512`, batch 16). Profile the GPU memory used during training:
1. Vanilla training. Note peak activation memory.
2. Enable gradient checkpointing on every block. Note peak activation memory and step time.

Expect roughly 4–6× memory reduction and 1.3× slower step time. Then push the batch size up until you're back at the original memory limit — the throughput-per-step is similar, but per-GPU you're now doing more samples per step. This is the practical win.

This exercise is worked end-to-end in [`supplementary/04_gradient_checkpointing.ipynb`](supplementary/04_gradient_checkpointing.ipynb) — a runnable nanoGPT-style model with memory profiling, the batch-size scale-up, and a gradient-equality check.
