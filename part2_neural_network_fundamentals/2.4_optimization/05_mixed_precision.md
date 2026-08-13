# Mixed Precision: fp16, bf16, fp8, and Loss Scaling

The bytes-per-number you choose for training has dramatic consequences for memory, throughput, and stability. The dominant modern recipe for LLM training is **bfloat16 (bf16)** with selective fp32 cast — but to get here we went through a brief flirtation with fp16-with-loss-scaling that turned out to be a footgun, and we're now moving toward fp8 at the frontier. Worth understanding each.

## Float precision recap

A floating-point number has two parts:
- **Exponent**: encodes magnitude (range of representable values).
- **Mantissa**: encodes precision (resolution at a given magnitude).

| Format | Bits | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~`±10^38` | ~7 decimal digits |
| fp16 (IEEE half) | 16 | 5 | 10 | ~`±6.5 × 10^4` | ~3 decimal digits |
| bf16 (Brain float) | 16 | 8 | 7 | ~`±10^38` | ~2 decimal digits |
| fp8 (E4M3) | 8 | 4 | 3 | ~`±448` | ~1 decimal digit |
| fp8 (E5M2) | 8 | 5 | 2 | ~`±5.7 × 10^4` | <1 decimal digit |

**bf16's key property**: same exponent range as fp32 (`±10^38`), but only 7 bits of mantissa. Throws away precision, keeps range. This is exactly the right tradeoff for DL — gradients can have wide magnitude ranges (need range), but the *exact* value of any individual gradient matters less than its direction (precision is dispensable).

**fp16's flaw for DL**: only 5 exponent bits → max representable value ≈ 65504. Small gradients (`< 6e-5`) underflow to 0. Both endpoints are problematic for DL.

## The fp16 era and loss scaling

Before bf16 hardware was widespread (~2019), the only choice for half-precision training was fp16. To avoid underflow on small gradients, NVIDIA developed **loss scaling**:

1. Multiply the loss by a large factor `S` (e.g. `S = 2^16`) before backward.
2. Backward propagates gradients scaled by `S` — they now fit comfortably in fp16's range.
3. Before the optimizer step, unscale gradients by `1/S` to recover the true gradient.
4. The optimizer step happens at fp32 with the unscaled gradients.

This is **mixed precision** in the original sense: forward in fp16, gradients in fp16-scaled, optimizer in fp32.

PyTorch's `torch.cuda.amp.GradScaler` automates this. It dynamically adjusts `S` — increasing it when steps succeed, decreasing it when a NaN or Inf appears (indicating overflow).

**The footgun**: `GradScaler` only works because of the dynamic adjustment. If `S` is too high → gradient overflow → NaN → scaler halves `S` and skips the step. If too low → gradient underflow → silent zero gradients → no error, just no training. Most of the operational headaches in fp16 training came from these.

## Bfloat16: the fix

bf16 has fp32's exponent range. No underflow, no overflow under any sane circumstances. **No loss scaling needed.** The whole `GradScaler` apparatus becomes unnecessary.

What you give up: precision. bf16 has 7 mantissa bits vs fp16's 10. So you can represent fewer distinct values at a given magnitude. For DL this is fine — the optimizer's update is `m_t / √v_t · η`, which is robust to small precision errors. Empirically, bf16 training matches fp32 quality on every standard benchmark.

The modern recipe:
- **Activations and gradients in bf16.** ~half the memory of fp32.
- **Optimizer state (Adam's `m`, `v`) in fp32.** This is the most precision-sensitive part — EMA averaging benefits from fp32 precision.
- **Master weights in fp32.** Updates are applied to fp32 weights; the bf16 copies used in forward/backward are cast from the fp32 master each step.
- **Normalization (LayerNorm/RMSNorm) computed in fp32.** The variance computation needs the precision.

This recipe — called "mixed precision" by convention, though really it's "bf16 with fp32 for things that need it" — is the default for all modern LLM training. PyTorch's `torch.amp.autocast(dtype=torch.bfloat16)` handles the casts.

**Hardware support**: A100 / H100 / H200 (NVIDIA), TPU v3+, MI250+ (AMD) all have native bf16 matmul. CPU support is patchier but functional.

## When does bf16 not "just work"?

Three places to watch:

1. **Cumulative sums** (e.g. computing softmax over a long sequence): repeated additions in low precision can accumulate error. Cast to fp32 for the sum, cast back.
2. **Norm computations** (variance in LayerNorm/RMSNorm): cast to fp32 for the math, cast back. PyTorch's `nn.LayerNorm` does this automatically.
3. **Loss computation**: a final reduction across all tokens in the batch. Cast logits to fp32 before softmax + cross-entropy. PyTorch's `F.cross_entropy` accepts bf16 logits but internally casts.

These are exceptions, not the rule. Most layers train fine in pure bf16.

## fp8: the frontier

H100 hardware supports fp8 matmuls. The Hopper architecture introduced two fp8 formats:
- **E4M3**: 4 exponent bits, 3 mantissa bits, range `±448`, more precision.
- **E5M2**: 5 exponent bits, 2 mantissa bits, range `±57344`, more range.

The recipe (Per et al. 2022, Transformer Engine):
- **Forward**: E4M3 (more precision needed for activations).
- **Backward**: E5M2 (more range needed for gradients).
- **Per-tensor scaling factors**: each tensor has a scale that's dynamically adjusted to keep its values in the fp8 representable range. Computed during the previous step or layer.
- **Master weights, optimizer state**: fp32 as before.

The throughput win: fp8 matmul is roughly 2× faster than bf16 on H100, with comparable training quality. The memory win is real but smaller (most memory is in activations and optimizer state, not weights).

Currently used in production at frontier labs (Anthropic, OpenAI, DeepSeek-V3) and via NVIDIA's Transformer Engine library. Less common in open-source training, but adoption is growing. fp4 is on the horizon (Blackwell GPUs) but not yet standard.

## Memory math, briefly

For a 7B parameter model in different precisions:

| Component | fp32 | bf16 + fp32 master | fp8 + fp32 master |
|---|---|---|---|
| Parameters | 28 GB | 14 + 28 = 42 GB | 7 + 28 = 35 GB |
| Gradients | 28 GB | 14 GB | 7 GB |
| Adam state (m + v) | 56 GB | 56 GB (fp32) | 56 GB (fp32) |
| Activations (training) | ~varies | ~half of fp32 | ~quarter of fp32 |

Total fp32 training memory: ~`4 × params = 112 GB` plus activations, before optimizer state sharding.
Total bf16+fp32 training: ~`4 × params = 112 GB` plus halved activations. (The master weights kill some of the win.)

With ZeRO-3 (sharded optimizer state across DP ranks), the optimizer state memory shrinks by `1/dp_size`, making the bf16 win much bigger in practice.

## Inference vs. training precision

Inference uses *less* precision than training because:
- No gradients → no underflow concerns.
- No optimizer state → simpler memory picture.
- Latency / throughput matters → push precision down for speed.

Common inference recipes:
- **bf16** or **fp16**: the "safe" inference choice. Matches training quality.
- **int8** (quantized weights): 2× memory savings over bf16, small quality loss. Standard for most production deployments.
- **int4** (GPTQ, AWQ): 4× memory savings. Noticeable quality loss but acceptable for many use cases. The "consumer-GPU LLM" precision.
- **fp8** on H100: faster than bf16 with negligible quality loss.

Quantization for inference is a deep topic — see Part 9.2 for the full treatment.

## A practical pitfall: `model.half()` vs `autocast`

Two ways to use half-precision in PyTorch:

```python
# 1. Cast the entire model to half precision.
model = model.half()                       # converts all params to fp16
out = model(x.half())                      # runs entirely in fp16

# 2. Use autocast (the modern way).
model = model.float()                      # keep params in fp32
with autocast(dtype=torch.bfloat16):
    out = model(x)                          # forward runs in bf16, params still fp32 (cast per op)
```

Approach 1 (`.half()`) converts everything, including norm parameters and the optimizer's master copy, to fp16. This is what NaN'd half-precision training in the early days. **Don't do this for training**.

Approach 2 (`autocast`) keeps master weights in fp32, casts on the fly for matmuls, keeps the precision-sensitive ops in fp32. This is the right pattern.

## Self-check

1. Why is bf16 strictly better than fp16 for DL training, despite having *fewer* mantissa bits? Be specific about what fp16 fails at.
2. What is loss scaling, and why does bf16 not need it?
3. Mixed precision keeps the optimizer state in fp32. Why? What goes wrong if you keep `m_t` and `v_t` in bf16?

### Answers

1. fp16's exponent range maxes at ~65000 and minimum normal value is ~6e-5. Gradients in deep nets routinely span this range — at init they're at one magnitude, after warmup they're at another, and per-coordinate gradients in attention can be 1e-3 to 1e3 within the same step. A non-trivial fraction of fp16 gradients underflow to zero (silent loss) or overflow to inf (instant NaN). bf16 has fp32's full exponent range — `±10^38` — so no realistic gradient is ever out of range. The price is half the mantissa precision (7 vs. 10 bits), but DL optimizers are robust to that level of noise. The choice between "occasional silent gradient loss" and "slightly noisier gradients" is one-sided.
2. Loss scaling multiplies the loss by a large constant `S` before backward, so all gradients are scaled up by `S` during backprop, fitting comfortably in fp16's representable range. Before the optimizer step, gradients are scaled back down by `1/S`. The dynamic version (`GradScaler`) adjusts `S` based on whether overflow occurs. bf16 doesn't need this because its exponent range is already fp32-equivalent — gradients fit without scaling. The entire `GradScaler` machinery becomes a no-op. Operationally this simplifies training code substantially and removes a category of bugs.
3. Adam's `v_t` is an EMA of squared gradients: `v_t = β2 · v_{t-1} + (1 - β2) · g²`. With `β2 = 0.999`, each update contributes `0.001 · g²` — a tiny fraction. In bf16 (7 mantissa bits ≈ 0.8% precision), this update can be entirely below the precision threshold and round to zero. Over many steps, `v_t` drifts away from the true second moment, eventually becoming meaningless. Same issue for `m_t` to a lesser extent (β1 = 0.9 means each update is ~10%, easier to represent in low precision). The fix: keep optimizer state in fp32. The cost is small because optimizer state is `2 × params` regardless of forward-pass precision, and you can shard it across GPUs (ZeRO-2/3) to amortize.

## Exercise

In PyTorch, train a small Transformer twice with the same seed:
1. Pure fp32.
2. bf16 with autocast, fp32 master weights and optimizer state.

Compare:
- Final loss (should match within ~1% — bf16 doesn't hurt quality).
- Memory usage (bf16 should be roughly half, plus optimizer overhead).
- Throughput (bf16 should be ~2× faster on bf16-native hardware).

Then try fp16 with `GradScaler` and watch for "Gradient scaler decreased scale to ..." messages — those are the dynamic adjustment kicking in to avoid overflow. This is the operational story of fp16 training in one workflow.
