# From Checkpoint to Embedding Model

> Convention note: row-vector convention — `Y = X W`, `W ∈ R^(d_in × d_out)`. Builds directly on [01_sentence_representations_in_lms.md](01_sentence_representations_in_lms.md); assumes the InfoNCE material from [1.3 mutual information](../../part1_math_foundations/1.3_information_theory/02_mutual_information.md).

## The conversion in one diagram

```
checkpoint LM  →  (optionally drop the causal mask)  →  pool(H) ∈ R^D
              →  (optional linear projection)        →  L2-normalize
              →  e ∈ R^D with ‖e‖ = 1
similarity(a, b) = ⟨e_a, e_b⟩ / τ       trained with InfoNCE
```

Architecturally that's the whole model. No new stack, no decoder, no output vocabulary — the unembedding matrix `W_U` is simply discarded. The entire conversion is a *training* problem, not an architecture problem.

## What makes it easy

**1. The semantics is already paid for.** Pretraining forced the network to represent meaning — you cannot predict the next token of text you don't understand. The embedding training does not have to teach the model what "interest rate" means or that "kitten" relates to "cat"; it only has to make that knowledge *readable as geometry*. This is why the conversion costs a vanishing fraction of pretraining compute.

**2. The strongest evidence: SimCSE.** Take BERT, encode every sentence *twice* with different dropout masks, call the two encodings a positive pair, all other in-batch sentences negatives, run InfoNCE for a few epochs. No new data, no labels, no information the checkpoint didn't already have — and sentence-similarity correlation (STS Spearman) jumps from the ~50s to the mid-70s. The knowledge was there; only the cone-shaped geometry (anisotropy, file 01) was in the way. Contrastive loss fixes the geometry: its two implicit objectives are **alignment** (positives map close together) and **uniformity** (embeddings spread over the sphere) — and uniformity is precisely an anti-anisotropy force.

**3. Fine-tuning converges absurdly fast.** E5-Mistral reached state-of-the-art with roughly **~1–2k contrastive steps of LoRA fine-tuning** on a Mistral-7B checkpoint. Compare: training a comparable-quality embedding model from random init is not merely slower — it is essentially impossible at that data scale, because the model would have to learn language itself from positive pairs alone.

**4. Everything in the LM toolbox transfers.** LoRA (keeps the base generative model intact — the adapter *is* the embedding model), longer-context bases give longer-context embedders, multilingual bases give multilingual embedders, and instruction-following ability gets reused for task conditioning (below).

## What makes it hard

**1. Objective mismatch — the LM never compared two texts.** Next-token prediction is a *within-sequence* objective; no part of pretraining ever pushed two related sequences together in representation space. Cosine distances on a raw checkpoint are uncalibrated leftovers, not trained quantities. A new training signal is unavoidable.

**2. The causal mask (file 01).** Last-token pooling works but is fragile — one position carries everything, and early-token information must survive a long attention relay. The three standard fixes:
- **Just unmask:** flip causal attention to bidirectional and let contrastive training adapt the weights. LLM2Vec adds a short adaptation stage (masked-token prediction with the mask off) before contrastive tuning so the weights aren't shocked by attention patterns they never saw.
- **Echo embeddings:** feed the text *twice* ("repeat the input: X ... X") and pool over the second copy — every token of the repeat has the full first copy in its prefix, bidirectionality smuggled through the causal mask at 2× cost.
- **Accept it** and rely on last-token pooling + instructions (E5-Mistral) — works better than it has any right to, since attention layers are an efficient information relay.

**3. Data is the actual moat.** The loss is a solved problem; the pairs are not. You need: positives that genuinely mean "same/relevant," **hard negatives** (lexically close but wrong — without them the model learns topic overlap, not meaning), and coverage of every intended task type. In-batch negatives come free but are mostly easy; false negatives (an in-batch "negative" that is actually relevant) poison training at scale.

**4. Asymmetric tasks.** A 6-word query and a 500-word passage that *answers* it should embed close — but "similar text" and "answers this question" are different relations. Solution: **instruction prefixes** — prepend a task description ("Given a web search query, retrieve relevant passages...") so one model serves retrieval, STS, classification, and clustering by conditioning the embedding on the relation wanted. This reuses the checkpoint's instruction-following — an LLM-base advantage BERT never had.

**5. You lose the generative model.** Full-weight contrastive tuning destroys generation quality — irrelevant if you ship a dedicated embedder, but it is why LoRA is popular here (base stays intact) and why "one model that both chats and embeds" remains mostly a research goal.

**6. Operational drag.** A 7B-LM embedder emits `D = 4096` floats per chunk — 4× the vector-DB cost of a BERT-scale 1024. **Matryoshka representation learning** (apply the loss to nested prefixes of the vector: first 256 dims, first 512, ...) trains embeddings that can be truncated at serve time with graceful degradation.

## The training recipe

The modern pipeline is two-stage contrastive, with the same loss and shifting data:

**Loss.** InfoNCE over a batch of pairs `(q_i, p_i)`, exactly the machinery from 1.3:

```
L = -mean_i log[ exp(⟨e_qi, e_pi⟩/τ) / Σ_j exp(⟨e_qi, e_pj⟩/τ) ]
```

with `j` ranging over the in-batch passages plus any mined hard negatives. Embeddings are L2-normalized, so scores are cosines; `τ` is small — typically **0.01–0.05** for text embeddings (much sharper than CLIP's learned ~0.07, because text positives are less ambiguous than image–caption pairs).

**Stage 1 — weakly-supervised contrastive pretraining.** Hundreds of millions of *naturally occurring* noisy pairs: title ↔ body, question ↔ accepted answer, citation contexts, translation pairs, anchor text ↔ page (E5's "CCPairs", GTE's mixture). No human labels. Batches are huge — tens of thousands — because with only easy in-batch negatives, the InfoNCE bound (1.3) needs `N` large to say anything. This stage teaches the broad relation "these co-occur for a reason."

**Stage 2 — supervised fine-tuning with hard negatives.** A much smaller curated mixture (MS MARCO, NLI, NQ, fact-verification, ...) where each positive comes with **mined hard negatives** — top-ranked non-answers retrieved by an earlier model, optionally filtered or relabeled by a cross-encoder/LLM to remove false negatives. Batches shrink; per-example difficulty replaces batch size as the source of signal. Instruction prefixes are attached per dataset. Optionally distill from a cross-encoder reranker (soft scores instead of hard labels), which is how several BGE variants squeeze out the last points.

**The lineage in one table:**

| Era | Model(s) | Backbone | Key move |
|---|---|---|---|
| 2019 | Sentence-BERT | BERT (encoder) | Siamese mean-pooled BERT + NLI labels — first practical sentence embedder |
| 2021 | SimCSE | BERT | Dropout-noise positives — proved it's geometry calibration, not new knowledge |
| 2022–23 | E5, BGE, GTE | BERT-scale encoders | Two-stage recipe: web-scale weak pairs → curated + hard negatives; instructions |
| 2023–24 | E5-Mistral | Mistral-7B (decoder) | LoRA + last-token pooling + LLM-generated synthetic pairs; ~1k steps to SOTA |
| 2024 | LLM2Vec | Llama-class | Unmask to bidirectional + brief adaptation + SimCSE — recipe, not a model |
| 2024–25 | NV-Embed, Qwen3-Embedding, gemma-embedding | 7B-class decoders | Learned (latent-attention) pooling, two-stage instruction-tuned contrastive, Matryoshka dims — current MTEB frontier |

The through-line: the *loss* hasn't changed since 2019. What changed is the backbone (encoder → decoder LLM), the *data* (labeled pairs → web-scale weak pairs + synthetic LLM-generated pairs + mined hard negatives), and the *conditioning* (none → per-task instructions).

## Why it matters in modern LLM work

Embedding quality upper-bounds every RAG system: retrieval failures are unrecoverable downstream no matter how good the generator is. The checkpoint-conversion view also explains the economics of the embedding-model market — anyone with a good open checkpoint and a strong pair-mining pipeline can produce a competitive embedder for a few thousand GPU-hours, which is why MTEB's top-20 churns monthly while the underlying recipe barely moves.

## Self-check

1. Why does SimCSE's dropout trick — positives carrying literally zero new information — improve sentence similarity so much? What does that imply about where the difficulty of building an embedder lies?
2. Stage 1 wants batch sizes in the tens of thousands; stage 2 works with small batches. What replaces batch size as the source of training signal, and why is it information-theoretically a fair trade?
3. You convert a causal 7B checkpoint with last-token pooling and it underperforms on long documents specifically. Give two mechanistic explanations and one mitigation for each.
4. Why are instruction prefixes more than a convenience — what failure mode of a single unconditioned embedding space do they fix?

### Answers

1. The checkpoint already encodes the semantics; what's broken is the geometry — anisotropy buries similarity signal in low-variance directions and the cone makes all cosines high. InfoNCE's uniformity pressure spreads embeddings over the sphere and its alignment pressure ties same-meaning texts together, so even information-free positives recalibrate the space. Implication: the hard part of embedding models is not "teaching meaning" (pretraining did that) but **data engineering for the contrastive signal** — which is exactly where stage-2 effort concentrates.
2. **Hard negatives.** InfoNCE's bound tightens with the number of negatives, but what actually matters is how much the negatives *constrain* the embedding. Ten thousand random negatives are mostly trivially distinguishable (different topics); one mined hard negative — same topic, wrong answer — forces a fine-grained distinction a random batch almost never produces. Per-example difficulty substitutes for count: you replace many weak constraints with few strong ones.
3. (a) **Single-position bottleneck:** everything must flow through one last-token state via the attention relay; over thousands of tokens, early-content recall degrades. Mitigations: learned/latent-attention pooling over all positions, or unmask + mean pooling (LLM2Vec-style). (b) **Length-induced distribution shift:** contrastive pairs are mostly short; positional behavior at long lengths is under-trained for the *embedding* task even if the base LM handles the length. Mitigations: include long-document pairs in stage 2, or chunk-and-aggregate at serve time. (Also acceptable: anisotropy interacts with length — more tokens pull pooled vectors toward the frequent-token cone; instructions/pooling changes mitigate.)
4. One unconditioned space must impose a single similarity relation, but "is a paraphrase of," "answers," and "belongs to the same cluster as" rank the same candidates differently — optimizing one relation actively hurts another (a question and its answer are *not* paraphrases). Instructions condition the embedding function on the relation, letting one set of weights realize several different similarity geometries instead of an averaged compromise of all of them.

## Exercise

Extend the layer-sweep script from file 01 into a minimal SimCSE run. Take GPT-2 or Qwen-0.5B, a few thousand unlabeled sentences (any corpus), and train for ~200–500 steps: encode each sentence twice with dropout active, last-token pooling, L2-normalize, InfoNCE with `τ = 0.05`, in-batch negatives, batch 64, LoRA or full fine-tune. Re-run the file-01 measurement (paraphrase-vs-unrelated cosine gap, per layer) before and after. You should see three things: average pairwise cosine of *unrelated* sentences drops sharply (uniformity fighting anisotropy), the paraphrase/unrelated gap widens severalfold (alignment), and the best layer migrates toward the top of the stack — the network reorganizes late layers to serve the pooled vector once the unembedding no longer claims them.
