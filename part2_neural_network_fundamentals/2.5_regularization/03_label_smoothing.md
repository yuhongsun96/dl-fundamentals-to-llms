# Label Smoothing

A small modification to the cross-entropy loss target: instead of training to predict probability 1 for the correct class and 0 for all others, train to predict slightly less than 1 for the correct class and slightly more than 0 for the others.

Common, simple, occasionally controversial. Used by the original "Attention Is All You Need" Transformer; mixed adoption in modern LLM pretraining.

## What it does

Standard cross-entropy targets a **one-hot** distribution:
```
y_true_i = 1 if i = correct_class else 0
```

The loss is `-log p_correct`.

Label-smoothed cross-entropy targets a **smoothed** distribution:
```
y_smooth_i = (1 - ε) if i = correct_class else ε / (V - 1)
```

where `V` is the vocab size and `ε` is the smoothing strength (typically `0.1`). For `V = 32000`, `ε = 0.1`:
- Correct class target: `0.9`.
- Each wrong class target: `0.1 / 31999 ≈ 3e-6`.

The loss is `-Σ_i y_smooth_i · log p_i = (1 - ε) · (-log p_correct) + ε · (-mean over wrong classes of log p_i)`. The first term is the standard CE loss with a tiny shrinkage; the second term is a uniformity penalty across all wrong classes.

## What problem it solves

The standard CE-with-softmax loss is **unbounded** — to drive `p_correct → 1`, you need `logit_correct → ∞`. The model is constantly incentivized to push the correct logit higher, with no asymptote, with no point at which it's "done." This causes:

- **Logit magnitudes grow without bound during training.** The pre-softmax logit for the correct class can hit 30+ by the end of training. Adjacent wrong-class logits become "approximately negative infinity" — the model is extremely confident.
- **Poor calibration.** A confidently-correct prediction at probability `0.9999` is essentially indistinguishable from `0.9` from a downstream perspective. But the loss treats them very differently. The model spends optimizer steps moving from `0.9999` to `0.99999` rather than learning genuinely uncertain regions of the task.
- **Reduced gradient signal on correct predictions.** Once the model is confidently correct (say `p_correct = 0.99`), the gradient is tiny. With label smoothing, even confidently-correct predictions retain some gradient because the target is `0.9`, not `1.0`.

Label smoothing puts a soft asymptote on confidence: the *best* you can do is `0.9` (or whatever the smoothed target is). The model isn't pushed past it, logits stay smaller, calibration is better, gradients remain useful longer.

## Where it helps

- **Translation (original use case)**: the 2017 Transformer paper used `ε = 0.1`. Helped BLEU score.
- **ImageNet classification**: standard for ResNet-era training. Helps top-1 accuracy slightly.
- **Speech recognition**: common in CTC/RNN-T training.
- **LLM SFT / chat fine-tuning**: occasional, but inconsistent. Some recipes use it, most don't.

## Where it doesn't help (or hurts)

- **LLM pretraining**: rarely used. The original GPT-2/3 didn't use it. Llama doesn't. Most modern open LLMs don't. Reasons:
  - Pretraining is single-epoch over web-scale data. Calibration concerns less acute when the model has barely seen the data.
  - Many "correct" continuations exist for any prefix — the per-token loss is already noisy in a way that label smoothing's noise doesn't add to constructively.
  - Modern eval is dominated by downstream benchmarks (MMLU, etc.) rather than perplexity, and label smoothing slightly worsens perplexity while not necessarily improving downstream scores.
- **Calibration-critical applications** that already have a separate calibration step (e.g. Platt scaling, temperature scaling at inference time): label smoothing is redundant.

## How it changes the gradient

Standard CE gradient (file 1.2/03): `∂L/∂logits = softmax(logits) - y_true = p - one_hot`.

Label-smoothed CE gradient: `∂L/∂logits = p - y_smooth = p - ((1-ε) · one_hot + ε / V · ones)`. For the correct class: `p_correct - (1-ε)`. For wrong classes: `p_wrong - ε/V`.

Two changes:
- The correct-class gradient becomes zero when `p_correct = (1-ε) = 0.9`, not at `1.0`. The model is "satisfied" at 0.9.
- The wrong-class gradients become zero when `p_wrong = ε/V`, which is positive (not zero). The model is "satisfied" with a small probability mass on wrong classes.

Both effects: the gradient stops pushing in directions that would just increase confidence without changing the answer.

## A different framing: knowledge distillation

Label smoothing is a special case of training against a "softer" target distribution. The general form is **knowledge distillation** (Hinton et al., 2015): train a smaller student model against a larger teacher's output distribution, which is naturally smoother than one-hot ground truth.

Label smoothing = distillation with a uniform teacher. Real distillation uses a learned teacher that knows which wrong answers are "closer" to right (e.g. "cat" misclassified as "tiger" is closer than as "truck"). This carries more useful signal than uniform smoothing.

LLM distillation (Part 9.2) uses this principle to train smaller production models from larger teacher LLMs. The math is the same as label smoothing but with `y_smooth` replaced by `y_teacher`.

## When to consider it

Use label smoothing if:
- You're training a classification model with a small fixed number of classes (image classification, NER, etc.).
- You're seeing overconfidence in eval (high accuracy but wildly miscalibrated probabilities).
- You're training translation or summarization with cross-entropy and care about BLEU/ROUGE.

Skip it if:
- You're pretraining an LLM. Won't help; might hurt perplexity.
- You're doing RLHF or any RL-based training. The reward structure handles calibration differently; label smoothing in the SFT phase is mostly ignored downstream.
- You have a calibration step (Platt / temperature scaling) at inference time.

## A practical detail

In PyTorch:
```python
F.cross_entropy(logits, targets, label_smoothing=0.1)
```

That's the entire integration. The function internally combines hard targets with `ε`-smoothed soft targets and computes the right loss. No custom implementation needed.

## Self-check

1. Why does standard CE+softmax push logits to grow without bound during training? What does label smoothing do to this dynamic?
2. The gradient `∂L/∂logit_correct` is `p_correct - 1` for standard CE and `p_correct - (1-ε)` for label-smoothed CE. Walk through what each gradient does once `p_correct = 0.95`.
3. LLM pretraining usually skips label smoothing, but SFT for some chat models uses it. What's different about the two regimes?

### Answers

1. To push `p_correct → 1`, you need `logit_correct → ∞` (because softmax never quite hits 1). The gradient `p_correct - 1` is nonzero for any finite logit, so the optimizer keeps increasing `logit_correct`. There's no asymptote. Over millions of steps, logits drift to large values (30+ for confident predictions) and softmax becomes nearly one-hot. With label smoothing, the target is `(1-ε)`. The gradient `p_correct - (1-ε)` is *zero* when `p_correct = (1-ε)` — say `0.9`. Once the model is confidently-but-not-overconfidently correct, the gradient says "you're done, don't make this more extreme." Logits stop growing, calibration stays reasonable.
2. **Standard CE**: `p_correct = 0.95` → gradient = `0.95 - 1 = -0.05`. Update pushes the logit *up* by `0.05` (effectively), making `p_correct → 0.96, 0.97, ...`. Still nonzero gradient, still trying to push to 1. **Label-smoothed (ε=0.1)**: `p_correct = 0.95` → gradient = `0.95 - 0.9 = +0.05`. Sign is *opposite*: the model is *over*-confident, so the gradient pushes the logit *down*, returning `p_correct` toward 0.9. The model is now "regretting" being too confident. Equilibrium is at `p_correct = 0.9` — neither lower nor higher gets zero gradient. This is the asymptote that label smoothing provides.
3. Pretraining: trillions of tokens, single epoch, every token's "correct continuation" is debatable (many valid next tokens for most contexts). The supervision signal is intrinsically noisy and softer than one-hot already (you could argue the *natural* target distribution over next tokens is smooth). Label smoothing's effect is small relative to this noise. SFT for chat: fewer training tokens (millions, not billions), multi-epoch, more "obvious" right/wrong answers for each chat turn. Here label smoothing can prevent overconfidence on the SFT distribution (which might not match real-user prompts), preserving some uncertainty for downstream calibration in RLHF. Different regimes; different cost/benefit.

## Exercise

Train a small Transformer for translation (or any small classification task) twice:
1. Standard `F.cross_entropy(logits, targets)`.
2. `F.cross_entropy(logits, targets, label_smoothing=0.1)`.

Compare:
- Training loss curves — the LS version should plateau at higher loss (it's bounded above by 0).
- Test accuracy — should be similar, possibly slightly better for LS.
- Output logit distributions — the LS version should have logits in a much narrower range (e.g. correct-class logits at 5-10 instead of 30+).
- ECE (expected calibration error) — the LS version should have much lower ECE.

This makes the calibration story tangible: lower max confidence, better-calibrated probabilities, similar accuracy.
