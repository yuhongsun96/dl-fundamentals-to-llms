# Deep Learning Review Outline (NLP → LLM/VLM Track)

Tailored for a former NLP practitioner returning to the field. Assumes prior familiarity with RNNs, LSTMs, and BERT-era Transformers. Skips CNNs, classical ML regression, and tabular/vision-only content.

Top-level references: [NOTATION.md](NOTATION.md) (symbols and conventions) and [ARCHITECTURE.md](ARCHITECTURE.md) (a one-page map of a modern decoder-only LLM with every standard trick assembled — the target the curriculum builds toward).

> **Tooling side-track:** [numpy_pytorch_schedule.md](numpy_pytorch_schedule.md) is a separate, short (~1 week) plan to rebuild NumPy/PyTorch fluency — the libraries this curriculum's code is written in. It's optional and independent of the 12 parts; do it whenever the implementation, not the theory, is what's slowing you down.

---

## Part 1 — Mathematical Foundations

### 1.1 Linear Algebra Refresher
- Vectors, matrices, tensors; shapes and broadcasting semantics
- Matrix multiplication as composition of linear maps; batched matmul
- Inner products, norms (L1/L2/∞), cosine similarity
- Outer products and low-rank structure (why LoRA works)
- Eigenvalues/eigenvectors, SVD — intuition, not proofs
- Projections and orthogonality (useful for attention / embeddings)
- Rotations in `n` dimensions: planes not axes; the canonical form; what rigidity excludes (reflections, translations)
- Jacobians and the chain rule in matrix form

### 1.2 Calculus & Probability for DL
- Gradients, directional derivatives, the chain rule
- Jacobian-vector products (JVP) and vector-Jacobian products (VJP)
- Softmax and log-softmax; numerical stability (log-sum-exp trick)
- Cross-entropy and KL divergence; why CE is the right loss for LMs
- Entropy, perplexity, and their relationship
- Expectation, variance; reparameterization trick (comes back for RL/VAE)

### 1.3 Information-Theoretic Intuitions
- Why next-token prediction = compression = learning
- Mutual information (shows up in contrastive learning / CLIP)
- The "bits per byte" framing of LM evaluation

### 1.4 Optional Deeper Knowledge
*A vocabulary layer, not new theory — the math language used casually in explanations of modern architectures. Optional and non-linear; read a group when its terms are costing you.*

**A — Space and size**
- Dimension, span, and rank: subspaces, ambient vs. linear vs. intrinsic dimension, effective rank
- Energy and spectra: why squares, `‖A‖_F² = Σσᵢ²`, and what "90% of the energy" does *not* claim

**B — Forms and read-out**
- Bilinear and quadratic forms: attention as one `M = W_Q W_Kᵀ`, gauge freedom, curvature
- Linear read-out and identifiability: when a learned `W` can pull an added signal back apart

**C — Symmetry and rotation** *(read in order; the language of RoPE)*
- Complex eigenvalues and rotation planes: conjugate pairs, `λ = r·e^(iθ)`, why rotation planes are 2-D
- Groups and the rotation group: abelian, `O(n)` vs. `SO(n)`, why 2-D rotations commute and 3-D ones don't
- Generators and one-parameter families: skew-symmetric generators, the matrix exponential, and why RoPE uses 2-D pairs

**D — Signals and estimates**
- Frequency, phase, and periodicity: wavelength, aliasing/Nyquist, why frequency ladders never repeat
- Approximations and orders of magnitude: "to first order," `O(1/√D)`, reading log-log plots

---

## Part 2 — Neural Network Fundamentals

### 2.1 The MLP as a Universal Building Block
- Linear layer + nonlinearity; stacking depth vs. width tradeoffs
- Activation functions: ReLU → GELU → SwiGLU (why modern LLMs use gated variants)
- The role of bias terms; why many modern models drop them

### 2.2 Backpropagation, Deeply
- Forward pass / backward pass as computational graph traversal
- Manual backprop through a 2-layer MLP (do this on paper once)
- Autograd mechanics: how PyTorch builds and walks the graph
- Gradient checkpointing: trading compute for memory
- Common gradient pathologies: vanishing, exploding, dead ReLUs

### 2.3 Initialization & Normalization
- Why init matters: variance preservation across layers
- Xavier/Glorot vs. Kaiming/He; what modern LLMs actually use
- BatchNorm (historical) vs. LayerNorm vs. RMSNorm — why NLP moved to LN/RMSNorm
- Pre-norm vs. post-norm Transformer blocks; why pre-norm won for deep stacks

### 2.4 Optimization
- SGD → momentum → Adam → AdamW (decoupled weight decay)
- Learning rate schedules: warmup, cosine decay, WSD
- Gradient clipping: why it's non-optional for Transformers
- Loss landscapes, sharpness, and why large-batch training needs care
- Mixed-precision training: fp16 vs. bf16 vs. fp8; loss scaling

### 2.5 Regularization
- Dropout (and why it's largely disappeared from large LMs)
- Weight decay as implicit regularization
- Label smoothing
- Data augmentation's role (and lack thereof) in language

---

## Part 3 — Residual Connections & Deep Networks

The synthesis / bridge chapter: promotes the residual connection from "one of Part 2's tricks" to *the* organizing abstraction of the Transformer. Leans on Part 2 (2.2/05 gradient pathologies, 2.3/03 norm layers, 2.3/04 pre-vs-post-norm) for mechanics and spends its pages on the *why* and on material those files deferred.

### 3.1 The Skip Connection
- The degradation problem: deeper *plain* nets get worse on *training* error; identity is hard to learn explicitly; the Highway Networks / LSTM-gating lineage
- Identity shortcuts and the "gradient highway" view (`I + ∂f/∂h`, forward and backward as one picture)
- Residual nets as an **ensemble of shallow paths**: the unraveled view, effective depth, layer lesioning/reordering, stochastic depth / LayerDrop
- The residual stream as the central abstraction: read → compute → add; superposition; the logit-lens / interpretability tie-in (forward → Part 11)

### 3.2 Normalization Placement & Depth
- Pre-LN vs. post-LN as a *residual-stream* story ("don't normalize the highway"); why RMSNorm replaced LayerNorm (see 2.3/03)
- At-scale stabilizers at the norm/residual interface: DeepNorm, Sandwich-LN, QK-norm, logit soft-capping — and which instability each targets
- Scaling the residual stream: variance grows `~L`, the final norm, `1/√(2L)` output-projection init, LayerScale / ReZero (the Highway gate, resurrected)
- Depth vs. width: why residuals make 1000 layers trainable but frontier LLMs stop at ~tens-to-~100 (diminishing returns, sequential-latency vs. parallel-width, stream bandwidth)

---

## Part 4 — Sequence Modeling Lineage (Brief)

Quick review since you know this — focus on *why each died*.

### 4.1 RNNs and LSTMs
- Hidden state recurrence; BPTT
- Vanishing gradient → gating (LSTM/GRU)
- Why they lost: sequential compute, limited context, hard to scale

### 4.2 Seq2seq and the First Attention
- Encoder-decoder with a bottleneck
- Bahdanau/Luong attention as learned alignment
- The intuition that planted the seed for Transformers

---

## Part 5 — The Transformer, Rebuilt From Scratch

Assume you've seen this but rebuild the mental model.

### 5.1 Self-Attention Mechanics
- Q, K, V projections; why three separate matrices
- Scaled dot-product attention; the √d_k scaling rationale
- Attention as soft dictionary lookup / kernel smoothing
- Multi-head attention: what each head can specialize in
- Causal vs. bidirectional masking

### 5.2 The Full Block
- Attention + residual + norm + MLP + residual + norm
- Why the FFN is usually 4× the model dimension
- Gated FFNs (SwiGLU, GeGLU) and the 2/3 width compensation

### 5.3 Positional Information
- Sinusoidal (original) — why it was OK but not great
- Learned absolute positions (BERT/GPT-2)
- Relative positions (T5 bias, ALiBi)
- **RoPE** (rotary position embeddings) — the modern default; the complex-plane rotation view
- Position interpolation / YaRN / NTK-scaling for context extension

### 5.4 Tokenization
- Character / word / subword tradeoffs
- BPE, WordPiece, SentencePiece, Unigram LM
- Byte-level BPE (GPT-2 onward) and why it handles everything
- Tokenizer pathologies: glitch tokens, SolidGoldMagikarp, numbers
- Emerging: tokenizer-free / byte-level (ByT5, MambaByte, BLT)

---

## Part 6 — Pretraining Paradigms

### 6.1 Pretraining Objectives
- **Causal LM** (GPT family): the autoregressive factorization, loss at every position, reading the loss curve
- **Masked LM** (BERT family) and **denoising / span corruption** (T5, BART): what bidirectional buys and costs
- Why decoder-only causal LM won: supervision density, task unification, one-stack simplicity, serving economics — and where encoders still live

### 6.2 Data
- The corpus lineage (Common Crawl → C4 → Pile → RedPajama → FineWeb) and what "one epoch" means now (per-source effective epochs)
- Deduplication (exact / MinHash / substring) and decontamination as the same operation; memorization
- Quality filtering: heuristic → classifier-based → model-based (FineWeb-Edu, DCLM); filtering as a scaling-curve shifter
- Mixtures & mid-training: mixing ratios, domain upsampling (code, math), the stable→anneal staged run (MiniCPM, Llama 3), annealing-as-probe
- Synthetic data: textbook generation (Phi), rephrasing (WRAP), distillation; model collapse and the generate→verify→train loop
- The data wall: the token stock, multi-epoch training (~4 epochs nearly free, ~16 dead), and the pressure valves

### 6.3 Scaling Laws
- Kaplan et al. (2020): the original power laws, `C ≈ 6ND`, and the flaw that built GPT-3
- Chinchilla (Hoffmann et al., 2022): compute-optimal ~20 tokens/param; why "together" (`α ≈ β`), and the published fit that contradicts the headline (Besiroglu refit)
- Post-Chinchilla reality: inference-optimal overtraining (Llama-3-8B at 1875 tokens/param); scaling laws as methodology (data, RL, inference-time axes)
- Emergent abilities — the mirage rebuttal, what survives it, and forecasting on smooth metrics

### 6.4 Training Mechanics at Scale
- Batches measured in tokens; steps = budget ÷ batch; critical batch size and batch-size warmup
- Sequence packing, document boundaries and intra-document attention masking, position-ID resets, loss masking (why `<eos>` gets loss)

---

## Part 7 — Modern LLM Architecture Refinements

*Quantitative throughout — the organizing fact is that memory bandwidth, not FLOPs, binds at inference (decode intensity ≈ 1 FLOP/byte vs. an H100's ~296).*

### 7.1 Attention Variants
- The KV-cache problem: bytes/token formula, why it overtakes the weights (70B @ 128K = 40 GiB/sequence), memory-bound decode
- Multi-Head → Multi-Query (MQA) → Grouped-Query (GQA); why `n_kv = 8` (quality knee *and* tensor-parallel alignment); uptraining
- Multi-head Latent Attention (MLA, DeepSeek): low-rank KV latent, matrix absorption, and the RoPE incompatibility forcing decoupled rotary keys
- Local and sparse patterns: sliding window (Mistral), local/global interleaving (Gemma), attention sinks, and 2025's learned sparsity (NSA)

### 7.2 Efficient Attention
- The memory-bandwidth wall: arithmetic intensity, roofline, prefill vs. decode as two different machines
- **FlashAttention** 1/2/3: online softmax derived, tiling + recomputation, why it's *exact*; what each version added
- KV-cache memory management: fragmentation, PagedAttention as virtual memory, prefix sharing / copy-on-write

### 7.3 Alternatives to Attention
- Linear attention: drop softmax → reassociate → a fixed-size recurrent state; why it lost in 2020 (state bottleneck, no forgetting, FlashAttention)
- State-space models: S4 → Mamba's selectivity (`Δ` as a forget gate) → Mamba-2's duality with linear attention
- Hybrids and the recall tradeoff: the pigeonhole ceiling on associative recall, and the field's ~1:7 attention ratio (Jamba, Zamba, Samba)

### 7.4 Mixture of Experts
- The sparse FFN: why the FFN, param-vs-FLOP accounting (DeepSeek-V3: 671B total / 37B active = 18× sparsity)
- Routing and load balancing: collapse as a feedback loop, aux loss, capacity factors, token- vs expert-choice, aux-loss-free bias control
- Fine-grained and shared experts: the combinatorial argument (`C(64,16)` vs `C(8,2)`), factoring out common knowledge, dense early layers
- The systems reality: expert parallelism, all-to-all as a barrier, memory cost, and the batch-size condition under which MoE stops paying

### 7.5 Long Context
- Where quadratic actually bites: attention is 6.5% of block FLOPs at 2K and 82% at 128K; the four distinct walls and their owners
- Training at long context: why data/tensor/pipeline parallelism all fail on one sequence; ring attention as distributed online softmax; the staged recipe (RoPE scaling itself is 5.3/05)
- Evaluating long context: why perplexity can't see it, NIAH's limits, RULER, and advertised vs. **effective** context

---

## Part 8 — Post-Training

### 8.1 Supervised Fine-Tuning (SFT)
- Instruction tuning; the FLAN / Natural Instructions lineage
- Chat templates and special tokens
- Data quality over quantity (LIMA hypothesis)

### 8.2 Preference Optimization
- **RLHF** pipeline: reward model + PPO
- Why PPO is finicky; the KL penalty's role
- **DPO**: derive the reward implicitly, skip the RL
- Variants: IPO, KTO, ORPO, SimPO — the tradeoffs
- Constitutional AI / RLAIF: synthetic preferences

### 8.3 Reasoning & RL
- Chain-of-thought as a prompting phenomenon
- Process vs. outcome reward models
- **RLVR** (verifiable rewards) — the DeepSeek-R1 / o1 paradigm
- GRPO and other critic-free RL methods
- Why "thinking" models changed the frontier

### 8.4 Parameter-Efficient Fine-Tuning
- LoRA: low-rank decomposition of weight deltas
- QLoRA: 4-bit base + LoRA adapters
- DoRA, VeRA, and the adapter zoo
- When to use full FT vs. LoRA

### 8.5 Embedding Models from LM Checkpoints
- What a checkpoint already gives you: contextual token vectors, no native sentence vector
- Pooling: CLS vs. mean vs. last-token; what the causal mask does to each
- Anisotropy — why raw LM cosine similarities are uncalibrated
- Contrastive adaptation: InfoNCE with in-batch + hard negatives
- The lineage: Sentence-BERT → SimCSE → E5/BGE/GTE → E5-Mistral / LLM2Vec / NV-Embed
- Two-stage recipe: weakly-supervised pairs at scale → curated fine-tune; instruction prefixes; Matryoshka embeddings

---

## Part 9 — Inference

### 9.1 Sampling
- Greedy, beam search (why beam search died for open-ended gen)
- Temperature, top-k, top-p (nucleus), min-p
- Repetition penalties and their failure modes

### 9.2 Inference Optimization
- KV caching: what's cached and why it dominates memory
- Prefill vs. decode phases; different bottlenecks (compute- vs. memory-bound)
- Continuous batching
- Speculative decoding (draft model + verify); Medusa, EAGLE
- Quantization: INT8, INT4, GPTQ, AWQ, GGUF, FP8
- Distillation for deployment

### 9.3 Structured Generation
- Constrained decoding, grammars, JSON mode
- Tool use / function calling as a decoding problem

---

## Part 10 — Multimodal: From CLIP to VLMs

### 10.1 CLIP and Contrastive Pretraining
- Dual-encoder architecture
- InfoNCE loss; temperature parameter
- Zero-shot classification via text prompts
- Limitations (bag-of-words behavior, compositionality failures)

### 10.2 Vision Transformers (just enough)
- Patch embeddings as tokenization for images
- ViT basics — you don't need CNN depth here
- Register tokens, SigLIP improvements

### 10.3 Vision-Language Models
- Architectural patterns:
  - Frozen encoder + projector + LLM (LLaVA-style)
  - Cross-attention fusion (Flamingo, Idefics)
  - Early-fusion / native multimodal (Chameleon, Gemini)
- The visual tokenizer / connector design space
- Training stages: alignment → instruction tuning
- Any-to-any models and unified tokenizers

### 10.4 Beyond Images
- Audio: Whisper's encoder-decoder, audio LMs
- Video: temporal tokenization challenges
- The "omni" trend (GPT-4o, Gemini)

---

## Part 11 — Evaluation & Interpretability

### 11.1 Evaluation
- Perplexity's strengths and deep limitations
- Benchmark landscape: MMLU, GSM8K, HumanEval, MATH, GPQA, SWE-bench
- Contamination, saturation, and why benchmarks rot
- LLM-as-judge: biases and how to mitigate them
- Arena-style human preference eval

### 11.2 Interpretability (optional but increasingly central)
- The residual stream view of Transformers
- Induction heads and in-context learning mechanics
- Sparse autoencoders and feature dictionaries
- Activation patching / causal tracing basics

---

## Part 12 — Systems & Scale (lighter pass)

- Data parallel, tensor parallel, pipeline parallel, sequence parallel
- ZeRO stages; FSDP
- Why training runs fail (hardware, loss spikes, data bugs)
- Rough cost/FLOPs intuition for a frontier-class run

---

## Suggested Pace

- **Weeks 1–2:** Parts 1–3 (math + NN fundamentals + residuals)
- **Week 3:** Part 4 (quick) + Part 5 (Transformer deep dive)
- **Weeks 4–5:** Parts 6–7 (pretraining + modern architecture)
- **Week 6:** Part 8 (post-training — this is where the field has moved most since BERT)
- **Week 7:** Part 9 (inference)
- **Week 8:** Parts 10–11 (VLMs + eval)
- **Week 9 (optional):** Part 12 (systems)

## Suggested Format Per Topic
1. Read one canonical paper or blog post
2. Implement the smallest possible version from scratch (nanoGPT-style)
3. Write a paragraph explaining it to yourself without notes

## Key Papers to Anchor On
- Attention Is All You Need (2017)
- BERT (2018) — you know this, skim
- GPT-2, GPT-3 (2019, 2020)
- Chinchilla (2022)
- LLaMA / LLaMA 2 / 3 papers
- FlashAttention (2022)
- RoPE (2021)
- InstructGPT / RLHF (2022)
- DPO (2023)
- DeepSeek-V3 and R1 (2024–2025)
- CLIP (2021), LLaVA (2023), Flamingo (2022)
