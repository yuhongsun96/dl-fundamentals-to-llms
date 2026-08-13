# NumPy & PyTorch — A Short Tooling Schedule

> **Note — this is a side-track, not part of the main curriculum.** [review_outline.md](review_outline.md) is the 12-part DL review; *this* file is a separate, deliberately short plan to rebuild fluency in the two libraries you'll implement everything in. It's tooling, not theory — the goal is to stop fighting the API so the concepts land. Written for a rusty practitioner who has used arrays before: it restores fluency, it doesn't teach programming.

**Timeframe:** ~1 week of evenings, or a focused long weekend. **5 sessions**, ~2–3 hours each. Ruthlessly scoped — everything here is load-bearing for LLM/Transformer work; the "deliberately omitted" list at the bottom says what was cut and why.

> **After these fundamentals, continue with the [PyTorch Model Tour](pytorch_model_tour/README.md)** — five short notebooks that train and evaluate small end-to-end models of increasing complexity (linear → MLP → embeddings → LSTM → self-attention), so you see real results and the architecture lineage in action.

**How to use:** each session is a **runnable notebook** — open it, run cells top to bottom, read each output, and change the numbers to see what moves. Type the exercises, don't just run them. The capstone (Session 5) is the payoff and the real test. The notebooks live in [numpy_pytorch/](numpy_pytorch/) and each runs end-to-end:

- Session 1 → [numpy_pytorch/01_numpy_core.ipynb](numpy_pytorch/01_numpy_core.ipynb)
- Session 2 → [numpy_pytorch/02_numpy_broadcasting_matmul_einsum.ipynb](numpy_pytorch/02_numpy_broadcasting_matmul_einsum.ipynb)
- Session 3 → [numpy_pytorch/03_pytorch_tensors_autograd.ipynb](numpy_pytorch/03_pytorch_tensors_autograd.ipynb)
- Session 4 → [numpy_pytorch/04_nn_module_training_loop.ipynb](numpy_pytorch/04_nn_module_training_loop.ipynb)
- Session 5 → [numpy_pytorch/05_capstone_transformer_block.ipynb](numpy_pytorch/05_capstone_transformer_block.ipynb)

The sections below are the at-a-glance plan (kept as prose because a checklist reads better flat); the notebooks are the actual study material.

---

## Session 1 — NumPy core: arrays, indexing, shapes

The mental model: an `ndarray` is a typed, N-dimensional block of numbers plus a `shape`. Everything is shape manipulation.

**Critical concepts**
- Creation & inspection: `np.array`, `zeros/ones/arange/linspace`, `.shape`, `.dtype`, `.ndim`.
- Indexing & slicing: basic slices, **boolean masks** (`x[x > 0]`), **fancy indexing** (`x[[0,2,5]]`).
- Reshaping: `reshape`, `.T`/`transpose`, `newaxis`/`None`, `squeeze`, `expand_dims` — and the difference between *reordering data* (`transpose`) and *reinterpreting the same buffer* (`reshape`).
- Reductions along axes: `sum/mean/max/argmax` with `axis=` and `keepdims=True`. Know *which axis disappears*.

**Don't skip:** `axis` semantics and `keepdims`. "Sum over the last axis" vs "over the batch axis" is the single most common source of silent bugs.

**Exercise:** given `x` of shape `(4, 5)`, compute per-row mean and std *without loops*, then standardize each row to mean-0/var-1. Confirm the result's per-row mean is ~0 (this is LayerNorm's core op).

---

## Session 2 — NumPy: broadcasting, matmul, einsum, vectorization

This session is where NumPy becomes powerful instead of just NumPy-with-loops.

**Critical concepts**
- **Broadcasting** — the rules (align trailing dims; size-1 or missing dims stretch). This is *the* NumPy superpower and its #1 footgun. Practice until `(B,S,D) + (D,)` and `(B,S,1) * (B,1,S)` are obvious.
- Matrix multiply: `@` / `np.matmul`, and **batched matmul** (leading dims broadcast, last two multiply) — the shape of every attention score computation.
- `np.einsum` — once, deliberately. `einsum('bsd,btd->bst', q, k)` *is* attention scores; being able to read/write einsum makes tensor code legible.
- Vectorization mindset: replace Python loops with array ops; know why (speed, and it's how the libraries are meant to be used).

**Don't skip:** broadcasting. Deliberately trigger a broadcasting bug (e.g. `(4,1)` vs `(1,4)` producing `(4,4)` when you wanted `(4,)`) so you recognize the smell.

**Exercise:** implement **softmax** over the last axis (with the max-subtraction trick for stability), then compute attention: given `Q,K,V` of shape `(S, d)`, produce `out = softmax(QKᵀ/√d) @ V`. Verify rows of the softmax sum to 1. (You've now written attention in raw NumPy.)

---

## Session 3 — PyTorch tensors + autograd

PyTorch tensors are NumPy arrays that (a) run on GPU and (b) track gradients. Almost every NumPy op has an identical-named torch op, so Sessions 1–2 transfer directly.

**Critical concepts**
- Tensors: `torch.tensor`, creation ops, `.shape`, `.dtype`, `.device`; NumPy interop (`torch.from_numpy`, `.numpy()`).
- **`view` vs `reshape` vs `.contiguous()`** — `view` needs contiguous memory, `reshape` falls back to a copy; know when you'll hit the error.
- `dtype` and `device`: `float32/bfloat16`, `.to(device)`, and picking `cuda`/`mps`/`cpu`.
- **Autograd** (ties directly to Part 2.2): `requires_grad=True`, build a computation by doing ops, `loss.backward()`, read `.grad`; `torch.no_grad()` for inference, `.detach()` to cut the graph, and why `.grad` **accumulates** (hence `zero_grad`).

**Don't skip:** autograd mechanics. Make a scalar `y` from a leaf tensor `x`, call `backward()`, inspect `x.grad`; call it twice without zeroing and watch the gradient double.

**Exercise:** re-implement Session 2's attention in PyTorch tensors, set `requires_grad=True` on `Q,K,V`, sum the output, call `backward()`, and confirm `Q.grad` is non-`None` with the right shape. Same math, now differentiable.

---

## Session 4 — nn.Module, layers, and the training loop

This is the session that turns "I can do tensor ops" into "I can train a model."

**Critical concepts**
- `nn.Module`: subclass it, register submodules in `__init__`, define `forward`; `.parameters()`, `nn.Parameter`, `.train()`/`.eval()`.
- The layers you'll actually use: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and `torch.nn.functional` (`F.softmax`, `F.cross_entropy`, `F.gelu`, `F.scaled_dot_product_attention`).
- **The canonical training loop** — memorize this rhythm:
  ```
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  optimizer.zero_grad()      # clear last step's grads
  loss.backward()            # populate .grad
  optimizer.step()           # update params
  ```
- `torch.optim.AdamW` (the optimizer from Part 2.4); `DataLoader`/`Dataset` (lightly — enough to batch data).

**Don't skip:** the `zero_grad → backward → step` order, and *why* `zero_grad` is there (gradients accumulate, Session 3). Forgetting it is the classic silent training bug.

**Exercise:** build a 2-layer MLP as an `nn.Module` for a toy classification task (random `(N, d)` inputs, integer labels), write the training loop, and watch the loss go down over ~100 steps. Then add `.to(device)` and run it on GPU/MPS if available.

---

## Session 5 — Capstone: a Transformer block from scratch

Everything above, assembled into the thing this whole repo is about. This is the test: if you can build and train this, you're fluent.

**Build (as `nn.Module`s):**
1. **Scaled dot-product attention** — from raw tensor ops (Session 2/3), *then* check your output against `F.scaled_dot_product_attention` on the same inputs (should match to float tolerance).
2. **Multi-head attention** — reshape `(B,S,D) → (B,H,S,D_h)`, batched attention, concat, output projection `W_O`. (Cross-check shapes against [ARCHITECTURE.md](ARCHITECTURE.md).)
3. **A pre-norm block** — `x = x + attn(norm(x)); x = x + mlp(norm(x))`, with a SwiGLU or GELU MLP.
4. Stack a few blocks + embedding + LM head into a tiny GPT and **train it** on a small char-level corpus with the Session 4 loop.

**Don't skip:** verifying your hand-written attention matches PyTorch's, and printing shapes at every step. **Shape-debugging is the real skill** — most PyTorch pain is a wrong `transpose` or a missing `keepdim`.

**Payoff:** this is exactly the nanoGPT-style model in the gradient-checkpointing walkthrough ([2.2/04 supplementary](part2_neural_network_fundamentals/2.2_backpropagation/supplementary/04_gradient_checkpointing.ipynb)) — you can now read and modify it line by line, and the rest of the curriculum's code is within reach.

---

## Critical things, in one checklist

If you internalize only these, you have the load-bearing 20%:

- [ ] **Broadcasting** rules (align trailing dims, size-1 stretches).
- [ ] **`axis` + `keepdims`** in reductions — know which axis disappears.
- [ ] **`reshape`/`view`/`transpose`/`contiguous`** — reorder vs reinterpret, and the contiguity error.
- [ ] **Batched matmul & einsum** — the shape of attention.
- [ ] **Autograd**: `requires_grad`, `backward`, `.grad` accumulates, `no_grad`, `detach`.
- [ ] **The training loop** and the `zero_grad → backward → step` order.
- [ ] **`nn.Module`** + `nn.Linear/Embedding/LayerNorm` + `F.cross_entropy`.
- [ ] **Shape-debugging** — print shapes; most bugs are shape bugs.

## What's deliberately omitted (to keep this short)

- **CNN/vision layers** (`Conv2d`, pooling) — out of this repo's scope (NLP→LLM).
- **Distributed training** (DDP/FSDP), custom CUDA, TorchScript/`compile`, profiling — real but Part 12 / production concerns, not fluency prerequisites.
- **The full DataLoader/Dataset ecosystem**, augmentation pipelines — learn just enough to batch; deepen when a project needs it.
- **NumPy numerics minutiae** (structured arrays, memmap, advanced dtypes) — rarely needed for DL.
- **Plotting** — pick up `matplotlib` basics ad hoc when an exercise wants a curve.
