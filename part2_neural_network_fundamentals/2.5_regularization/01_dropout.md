# Dropout — and Why It's Largely Disappeared from Large LMs

For a decade after Hinton et al. introduced it in 2012, dropout was the default regularizer in every deep learning recipe. In modern LLM pretraining it's almost entirely absent. Worth understanding what it does, why it worked so well, and why it stopped being needed.

## What dropout does

At every forward pass during training, randomly zero out each activation independently with probability `p` (the **dropout rate**), then scale the surviving activations by `1/(1-p)` so the expected value is preserved:
```
mask = Bernoulli(1-p, shape=x.shape)
y = (x ⊙ mask) / (1 - p)
```

At inference time, dropout is **off** — `y = x`, no masking. The scaling during training ensures forward-pass expectation matches inference behavior.

Typical `p`: 0.1 (BERT, GPT-2, T5) to 0.5 (small image classifiers).

## Why it was so effective

Three intuitions, all true to varying degrees:

1. **Implicit ensemble.** Each forward pass uses a random subset of activations — effectively training a different sub-network. At inference, using all activations is like ensembling over all `2^N` sub-networks. Ensembling reduces variance → better generalization.
2. **Prevents co-adaptation.** Without dropout, a unit can rely on a specific pattern of other units always being active. With dropout, no unit can count on its neighbors — each must do something useful on its own. Forces redundancy and distributed representations.
3. **Implicit regularization.** Random perturbations of activations bound the effective Lipschitz constant of the function, discouraging sharp dependence on any single input feature.

For the small-data, moderate-parameter regime of 2012–2018, this combination dramatically reduced overfitting. It was a free win — drop in a line of code, get +1–2% test accuracy.

## Where it's still used

- **Vision models on small datasets**: ImageNet-scale and below, particularly with limited augmentation. Still standard.
- **Fine-tuning on small task-specific datasets**: where overfitting is a real risk because the dataset is small relative to the model. BERT fine-tuning, for instance, uses dropout `p = 0.1`.
- **RNN training**: variational dropout (same mask reused across time steps) was important for LSTMs. Mostly historical now.
- **Some smaller LLMs and encoder models**: BERT, T5, OPT all have dropout enabled at `p = 0.1` during pretraining.

## Why modern LLM pretraining drops it

Several converging reasons:

### 1. Data abundance

LLM pretraining uses trillions of tokens. The model sees each parameter's "right" value through enormous data volume. Overfitting on the training data isn't the bottleneck — the bottleneck is **underfitting**, i.e. extracting maximum signal from the available compute. Anything that *slows learning* (which dropout does by injecting noise) directly trades signal for nothing.

The folk rule: dropout helps when `parameters >> data`. Modern LLM pretraining has `data >> parameters` for any individual epoch. Different regime.

### 2. Single-epoch training

LLM pretraining typically goes through the data **once or fewer** (Chinchilla-optimal training uses ~20 tokens per parameter, often less than one full epoch of the available data). With no repetition, the model can't memorize specific examples to overfit to. Dropout's anti-memorization effect has nothing to fight.

Compare to vision: ImageNet training does 90+ epochs over 1.3M images. Dropout's anti-memorization is meaningful there.

### 3. Modern architectural regularization

Other modern components implicitly regularize:
- Weight decay shrinks weights smoothly.
- LayerNorm/RMSNorm bound activations.
- Pre-norm stabilizes the residual stream.
- Mixed precision adds slight numerical noise.

Combined, these provide enough implicit regularization that dropout's explicit noise adds nothing.

### 4. Empirical evidence

GPT-3 and successors (Llama, Mistral, etc.) explicitly removed dropout during pretraining and reported either neutral or *positive* effects — i.e. removing it slightly improved final loss. The Chinchilla and PaLM papers report similar.

### 5. Compute cost

Dropout adds a small per-step compute cost (mask generation, multiply). At frontier scale, "tiny per-step cost over months of training" is non-trivial. Removing it is a free throughput win.

## Where dropout still appears in modern LLMs

- **Fine-tuning** (especially LoRA / SFT on smaller datasets): `p = 0.05` or `p = 0.1` is common. The smaller-data, multi-epoch regime is where dropout's anti-memorization is back in play.
- **Embedding dropout**: some models keep `p = 0.1` on the input embeddings only, as a mild regularizer. Inconsistent across implementations.
- **Attention dropout**: dropout on the attention weights after softmax. Used in some older Transformer variants; mostly absent in modern LLMs. When present, `p = 0.1`.

The Llama 2 paper explicitly states: no dropout during pretraining. Llama 3, Mistral, Qwen, DeepSeek: same. If you're reading a 2023+ LLM architecture paper, assume no dropout unless they call it out.

## A subtle issue: dropout's effect on attention

If you do enable attention dropout, it interacts with attention's existing mechanism in interesting ways. Attention softmax produces a probability distribution over keys; dropping out entries of this distribution means a query attends to a random subset of keys with each pass. This is structurally similar to "random key masking" — a different regularizer that has been studied independently.

In FlashAttention (Part 7.2), attention dropout is tricky to implement efficiently because the attention matrix is never fully materialized. Most modern Transformer implementations bypass this by simply not using attention dropout.

## Variants

A few extensions of dropout that show up in code:

- **DropConnect**: drop weights instead of activations. Rarely used in modern stacks.
- **Variational dropout** (Gal & Ghahramani, 2016): same mask across time steps in an RNN. Made LSTMs more trainable. Mostly historical.
- **Stochastic depth** (Huang et al., 2016): drop entire residual sublayers with some probability during training. Used in some vision Transformers (e.g. ViT, DeiT). Not used in LLM pretraining.
- **DropPath**: similar to stochastic depth, drops the output of a residual block to zero. Some text-vision joint models use it.

None of these have entered the standard LLM pretraining recipe.

## When to add dropout back

For fine-tuning a pretrained LLM on a small downstream task:
- Yes, add dropout. `p = 0.05` or `0.1`. The overfitting risk is real.

For pretraining from scratch on web-scale data:
- No. Trust the implicit regularization.

For mid-size training on a moderately-sized domain dataset (say, 1B tokens of domain-specific text):
- It's a coin flip. Sweep `p ∈ {0, 0.05, 0.1}` and pick the best. Often the answer is 0.

## Self-check

1. Dropout scales surviving activations by `1/(1-p)` during training. Why? What goes wrong if you skip the scaling?
2. Why does dropout slow learning, and why is this a price worth paying for small-data training but not for large-data LLM pretraining?
3. The Llama paper doesn't use dropout but does use weight decay. Aren't these both "regularizers"? Why use one and not the other?

### Answers

1. The scaling makes the expected forward output match the inference output. Without scaling: at training, `E[y] = (1-p) · x` (each entry has probability `1-p` of surviving). At inference, `y = x`. So the activations have *different magnitudes* at train and inference — the network sees one distribution during training and a different one during evaluation. With the `/(1-p)` scaling, `E[y] = x` at train, matching inference. The network's parameters can be tuned consistently. Without scaling, you'd have to undo the magnitude mismatch some other way (e.g. by scaling the weights at inference time — the "inverted dropout" trick), but the scale-at-train version is cleaner and is the standard.
2. Dropout zeroes a fraction of activations every step, which forces the model to relearn what it would have learned slightly faster without the noise. The gradients are noisier, the effective signal-to-noise ratio is reduced, and convergence is slower in wall-clock terms. In small-data regimes, the slowdown is worth it because the alternative is overfitting — converging fast to a solution that doesn't generalize. In large-data LLM pretraining, the model is already underfitting (compute-limited, not data-limited), so any slowdown directly costs you final quality. Dropout slowed pretraining without helping it; remove it and you train faster with equal or better final loss.
3. They regularize differently. Weight decay pulls all weights toward zero, providing a global complexity prior — *fewer effective parameters*. It costs nothing in training speed (small additive term per step) and has a clear theoretical motivation (Bayesian prior on parameters). Dropout perturbs activations, which slows learning. For LLM training, weight decay is free; dropout costs speed. The cost/benefit favors weight decay alone. They're not redundant either: weight decay constrains *parameters*, dropout perturbs *activations*. The Llama recipe says "we want the parameter prior, we don't want the activation perturbation." Different tools for different jobs.

## Exercise

Train two 6-layer Transformers on a moderate-size LM dataset (e.g. WikiText-103, ~100M tokens), 5 epochs each:
1. Without dropout.
2. With `p = 0.1` on activations and attention weights.

Track train loss and validation loss over epochs.

In the no-dropout run, train and val should diverge in later epochs (overfitting). In the dropout run, they should track each other better (less overfitting) but converge to a higher validation loss. This is dropout in its classical regime — useful when overfitting is a real risk.

Now repeat with only 1 epoch over a much larger dataset (e.g. RedPajama subset, ~10B tokens). In this regime, both runs should produce similar val loss, with the no-dropout version slightly better. This is the modern LLM regime — dropout no longer helps.
