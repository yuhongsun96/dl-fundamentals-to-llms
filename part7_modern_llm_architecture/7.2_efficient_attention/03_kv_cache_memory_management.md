# KV-Cache Memory Management

[7.1](../7.1_attention_variants/01_the_kv_cache_problem.md) shrank the cache architecturally; FlashAttention ([file 02](02_flashattention.md)) removed attention's *other* memory cost. What remains is a pure systems problem: given a cache you can't shrink further, how do you *store* it without wasting most of it? The answer — treat GPU memory like an operating system treats RAM — bought a 2–4× throughput improvement with no model change at all, which makes it one of the best returns on engineering in the stack.

**Scope:** the memory *layout* idea and what it enables. Scheduling policy, continuous batching, and the rest of serving live in Part 9.2.

## The problem: internal fragmentation

The obvious implementation reserves each sequence a contiguous buffer sized for the worst case:

```
allocate max_context × KV_bytes_per_token   per sequence slot
```

At `max_context = 128K` on a 70B model that's **40 GiB per slot** ([7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)). Now serve real traffic, where most requests are short: a 500-token conversation occupies a 128K-token slot and **wastes 99.6% of it.** You've sized memory for the longest request any user might send and paid that for every user.

The vLLM paper (Kwon et al., 2023) measured this on real systems and found only **20–40% of allocated KV memory actually held tokens.** Three distinct wastes:

- **Internal fragmentation** — the reserved-but-unused tail of each slot (the dominant one).
- **Reservation waste** — space held for tokens the sequence hasn't generated yet.
- **External fragmentation** — gaps between variable-sized contiguous blocks that no request fits into.

And by [7.1/01](../7.1_attention_variants/01_the_kv_cache_problem.md)'s logic, wasted KV memory converts directly into lost throughput: fewer concurrent sequences means lower batch size means a more memory-bound decode ([file 01](01_the_memory_bandwidth_wall.md)).

## PagedAttention: virtual memory for the cache

The fix is the classic OS solution, transplanted. Stop requiring contiguity:

- Split the cache into fixed-size **blocks** (typically 16 or 32 tokens' worth of K/V).
- Each sequence gets a **block table** mapping its logical token positions to arbitrary physical blocks.
- Blocks are allocated **on demand** as the sequence grows, from a global pool.
- The attention kernel is modified to gather K/V through the block table instead of assuming a contiguous stride.

The correspondence is exact and worth naming, because it's why the design is trustworthy: **blocks = pages, block table = page table, the sequence's token positions = virtual address space, the pool = physical memory.** Fragmentation drops to at most one partially-filled block per sequence — bounded waste of ≤ 16 tokens instead of unbounded waste of ≤ `max_context` tokens. Reported utilization goes from 20–40% to near-100%, and throughput improves 2–4× at equal hardware.

The cost is a level of indirection in the hottest kernel in the system, which is why this required a custom attention kernel rather than a memory allocator change — and why paged and non-paged kernels are separate code paths everywhere.

## The bonus: prefix sharing via copy-on-write

Once the cache is paged, sharing becomes nearly free — and this is where the OS analogy pays a second dividend. Two sequences with a common prefix can **point at the same physical blocks**:

- **System prompts and few-shot templates**: hundreds or thousands of tokens identical across every request in a deployment, now stored once instead of per request.
- **Parallel sampling** (`n` completions of one prompt): the prompt's blocks are shared by all `n`; only the divergent tails allocate.
- **Beam search / tree search**: branches share their common ancestry.

Writes trigger **copy-on-write** on the affected block only, exactly as in `fork()`. This is the mechanism behind "prompt caching" features in production APIs — a real cost reduction for the very common workload of a long fixed prefix plus a short variable suffix (which is the deployment shape Part 6's system-prompt discussion assumed).

Practical note: because sharing is block-granular, it only kicks in on *exact* prefix matches aligned to block boundaries — so a template that differs in its first token shares nothing. That granularity effect is worth knowing when designing prompts for cacheability.

## What this doesn't fix

Keeping the boundaries honest:

- **It doesn't reduce the cache's fundamental size** — that's the architecture's job ([7.1](../7.1_attention_variants/02_mqa_and_gqa.md)–[03](../7.1_attention_variants/03_multi_head_latent_attention.md)). Paging eliminates *waste*, not *requirement*.
- **It doesn't help a single long sequence.** One 128K request genuinely needs its 40 GiB; paging helps when you have *many, varied* requests.
- **It adds kernel complexity and a small indirection cost**, paid on every attention call.

The complementary technique, worth naming because it's the natural next lever: **KV-cache quantization** (int8 or fp8 keys and values) attacks the `bytes_per_value` term for another ~2× — with quality caveats, especially on keys, and it's a Part 9.2 topic.

## Why it matters in modern LLM work

- **It's why vLLM/TGI/SGLang exist and why you should use one** rather than a naive generation loop — most of the throughput gap is this.
- **Prompt caching is an API-visible feature** built on it; understanding block-granular prefix matching tells you how to structure prompts to hit it.
- **The general lesson transfers:** when a resource is allocated in variable-sized contiguous chunks with unpredictable lifetimes, paging is usually the answer. LLM serving rediscovered 1960s operating systems, and that's a compliment to both.

## Self-check

1. Name the three kinds of KV memory waste in a contiguous-allocation server, and which dominates.
2. Why does wasted KV memory reduce *throughput* rather than merely capacity? Trace the causal chain.
3. Give the full correspondence between PagedAttention and OS virtual memory (four terms).
4. What bounds fragmentation waste per sequence under paging, and how does that compare to the contiguous scheme?
5. Two API requests share a 900-token system prompt but the second adds one word to the *front*. How much block sharing occurs, and why?
6. Paging gives 2–4× throughput. Why doesn't it help a single 128K-token request at all?

### Answers

1. **Internal fragmentation** (the reserved-but-unused tail of each max-length slot) — dominant; **reservation waste** (space held for not-yet-generated tokens); **external fragmentation** (unusable gaps between variable-sized allocations). Measured utilization in contiguous systems was only 20–40%.
2. Wasted memory → fewer concurrent sequences fit → smaller decode batch → lower arithmetic intensity ([file 01](01_the_memory_bandwidth_wall.md), intensity ≈ batch size) → the GPU spends more time waiting on weight reads per token produced. Since decode needs `B ≈ 300` to approach compute-bound, anything that caps `B` caps throughput directly.
3. Cache blocks = **pages**; the per-sequence block table = **page table**; the sequence's logical token positions = **virtual address space**; the global block pool = **physical memory**. Copy-on-write for shared prefixes completes the analogy with `fork()`.
4. At most **one partially-filled block** — ≤ 16 or 32 tokens of waste per sequence. Contiguous allocation wastes up to `max_context − actual_length` tokens, i.e. potentially ~128K tokens (40 GiB on a 70B). Bounded-small versus unbounded-large is the entire improvement.
5. **None.** Sharing requires an exact prefix match from token 0, and prepending a word shifts every subsequent token's position, so no block's contents match. (Appending a suffix would share all complete leading blocks.) This is why cacheable prompt design puts the fixed content first and the variable content last, and aligns to block boundaries where possible.
6. Because that request's 40 GiB is a genuine requirement, not waste — its blocks are all full. Paging recovers *unused reserved* space, so its benefit scales with the variance in request lengths and the number of concurrent requests. One maximal request has neither.

## Exercise

Simulate the allocator — no GPU needed. (a) Generate 1,000 request lengths from a realistic skewed distribution (e.g. lognormal, median ~400 tokens, p99 ~30K, cap 128K). Using the 70B `320 KiB/token` figure, compute total memory needed under (i) contiguous allocation at `max_context = 128K`, (ii) paging with 16-token blocks. Report the utilization ratio for each and compare against the paper's 20–40% claim. (b) For a fixed 600 GiB budget, compute how many concurrent requests each scheme admits, and convert to a throughput ratio using intensity ≈ batch size. (c) Add a shared 1,000-token system prompt to every request and recompute the paged number with prefix sharing — quantify the additional win. (d) One sentence: at what request-length distribution would paging stop mattering?
