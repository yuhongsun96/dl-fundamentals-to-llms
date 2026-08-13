# Encoder–Decoder with a Bottleneck

Seq2seq (Sutskever et al., 2014; Cho et al., 2014) was the architecture that made neural machine translation work — and it contained one flaw so specific and so fixable that fixing it produced attention (file `02`) and, three years later, the Transformer. This file isolates that flaw.

**Convention:** column-vector; encoder/decoder hidden states `∈ R^D`.

## The architecture

Two RNNs, glued at a single vector:

```
Encoder:   reads source x_1…x_S,  produces states  s_1…s_S
           context c := s_S                          ← the LAST encoder state, one vector
Decoder:   initialized from c,  generates target y_1…y_T  one token at a time,
           each step conditioned on c and the tokens so far
```

The encoder compresses the *entire* source sentence into a single fixed-length vector `c` (the final hidden state). The decoder then unrolls a language model from `c`. It's elegant — arbitrary-length input to arbitrary-length output, fully differentiable, end-to-end — and for short sentences it worked well enough to beat phrase-based statistical MT.

## The bottleneck

Everything the decoder will ever know about the source must pass through `c`, a fixed `D`-dim vector, **regardless of source length**. This is the file-`01`/`4.1` fixed-state ceiling, now in its most acute form: it's not just that history is compressed, it's that a whole 40-word sentence and a whole 4-word sentence get the *same* `D` numbers of description.

The symptom was unmistakable and became the field's motivating plot: **translation quality falls off a cliff as source sentences get longer.** Short sentences: fine. Long sentences: the single vector can't hold enough, early-source information is overwritten by the time encoding finishes, and the decoder produces fluent-but-unfaithful output. Practitioners even resorted to **reversing the source sequence** (Sutskever's trick) so the first source words — often aligned to the first target words — sat closer in the unrolled graph to where decoding starts. That such a hack helped at all is a tell: the model's access to the source was distance-limited and bottlenecked, exactly the disease.

## Why it matters in modern LLM work

- **It names the anti-pattern attention exists to kill.** "Force all of X through one fixed vector, then read from that vector" is the bottleneck; "keep all of X and let the reader attend to the relevant parts" is the fix (file `02`). Once you've felt this failure, scaled dot-product attention (Part 5) reads as the obvious repair rather than a magic trick.
- **The encoder–decoder *shape* survived; the bottleneck didn't.** Cross-attention (Part 5) keeps "encode the source, decode the target" but replaces the single `c` with attention over *all* encoder states. Encoder–decoder Transformers (T5, BART; the original 2017 model) are this. Decoder-only LLMs (Part 6.1) later dropped the split entirely — but the pattern still shows up wherever one modality conditions another (e.g., cross-attention VLMs, Part 10.3).
- **It's the cleanest illustration of "compression vs. direct access."** The seq2seq bottleneck is the RNN state ceiling (Part 4.1) taken to its logical extreme — one vector for the whole input — which makes it the perfect foil for understanding why attention's refuse-to-compress design wins on quality (at the cost of memory, Part 9.2).

## Self-check

1. In vanilla seq2seq, what exactly is the context `c`, and what is its size as a function of source length `S`?
2. Translation quality dropped sharply for long sentences. Tie that symptom to the specific property of `c`.
3. Reversing the source sequence improved results. What does the fact that this hack helped reveal about the model's weakness?
4. Attention (file `02`) keeps the encoder–decoder shape but changes one thing. What does it replace, and what does the decoder get access to instead?

### Answers

1. `c` is the encoder's *final* hidden state `s_S` — a single fixed `D`-dim vector. Its size is `D` **regardless of `S`**: a 5-word and a 50-word source both get exactly `D` numbers.
2. Because `c` is fixed-size, longer sources must cram more information into the same `D` dimensions; capacity is exceeded, early-source content is overwritten during encoding, and the decoder lacks faithful detail — so quality degrades with length.
3. That the model's effective access to source information was **distance/position-limited** — putting source words nearer (in the unrolled graph) to where the decoder needs them helped, which only matters if the single-vector channel couldn't carry information reliably across long spans. A symptom of the bottleneck, patched positionally.
4. It replaces the single fixed context `c` with a *dynamic* context computed at each decoder step as a weighted combination of **all** encoder states `s_1…s_S`. The decoder gets direct, selective access to the entire source instead of one compressed summary.

## Exercise

Estimate the bottleneck quantitatively. Suppose `D = 512` (fp32) and you translate a 50-token source where each token's "useful" content is ~roughly a `D`-dim vector's worth. (a) How many `D`-vectors of source information does attention (file `02`) keep available to the decoder, versus vanilla seq2seq? (b) In one sentence, argue why no amount of increasing `D` cleanly fixes seq2seq the way attention does. (c) Name the resource attention spends to buy this (hint: it reappears as a Part 9 inference cost) and why that trade was worth it for quality.
