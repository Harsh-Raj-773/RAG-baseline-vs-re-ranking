# Report scaffold (replace each "YOUR WORDS" block, then delete this line)

> The assignment says the report must be in your own words and not AI-generated.
> This file only collects the **facts and numbers from the notebook** under the four required headings.
> Write the explanations yourself. Keep it concise (about 2 pages).

**Setup in one line:** DPDP Act 2023 (21-page Gazette PDF) → 92 section-aware chunks → bi-encoder `all-MiniLM-L6-v2` + FAISS (top 10)
→ optional cross-encoder `ms-marco-MiniLM-L-6-v2` re-rank → top 3 → Qwen2.5-3B-Instruct (local, greedy, same prompt).

---

## 1. Architecture comparison

YOUR WORDS. Points to cover:
- what the bi-encoder encodes and *when* (chunks offline, query at run time; one vector each; similarity = cosine)
- what the cross-encoder sees (query and chunk in one input; attention between their tokens; outputs one score)
- what each one can and cannot capture (e.g. "prohibited" vs "shall not", an exception phrased differently)
- why one can be indexed and the other cannot

## 2. Qualitative & rank-shift analysis

**Rank shifts of the failure queries** (gold chunk rank, bi-encoder → cross-encoder):

| Query | Category | Gold chunk | Rank shift | Answer 1 | Answer 2 |
|---|---|---|---|---|---|
| N3: What is the Board *prohibited* from doing when it inquires into a company? | negation | 63 (s.28(8)) | #4 → #1 | refusal | correct |
| E6: Does the Act apply to data used for research or statistics? | exception | 45 (s.17(2)(b)) | #4 → #1 | refusal | partial (cites exemption, says Act "applies") |
| K6: From what age is a person no longer a child? | keyword trap | 2 (s.2(f)) | #7 → #1 | refusal | wrong (says no age given) |
| K1: Maximum fine for failing to protect data from a breach? | keyword trap | 90 (Schedule) | #1 → #1 | **wrong: Rs 200 crore** | correct: Rs 250 crore |
| B5: Must I approach the company before the Board? | buried | 38 (s.13(3)) | >10 → >10 | **hallucination** | **hallucination** |
| B1: Penalty for impersonation? | buried / multi-hop | 91 (Schedule) | >10 → >10 | partial | partial |

Overall (24 queries): Hit@3 0.667 → 0.792, MRR@10 0.589 → 0.743, Recall@10 0.792 → 0.792 (unchanged, as it must be).
3 fixed, 0 hurt, 16 unchanged, 5 out of reach. Answer labels: correct 12 → 11, partial 4 → 8, refusal 6 → 3, wrong 2 → 2.

Figures: `artifacts/figures/rank_shift_all.png`, `artifacts/figures/bump_charts.png`. Full tables: notebook Sections 8–9.

YOUR WORDS. Points to cover:
- why the baseline ranked the distractors higher in N3 / E6 / K6 (look at the top-3 chunk headers in the notebook's candidate tables)
- Answer 1 vs Answer 2 for N3 (refusal → correct) and K1 (misstated amount → correct): what did the baseline context lack or add?
- the honest part: re-ranking fixed retrieval, but E6 and K6 answers were still flawed. What does this say about the 3B LLM vs the retriever?
- why correct-answer count did not rise overall (K4, E2, E7, N1 went correct → partial even though the gold chunk was #1 in both; the other two chunks changed)
- the recall ceiling: B3 is at rank 11, B4 at 29, B5 at 31, N6 at 50, B1 at 53. Re-ranking top-30 would recover B3 and B4 (notebook cell after 9.6). B5 shows a hallucination caused by a missing chunk.

## 3. Latency profiling

Median over 24 queries × 5 repeats, RTX 4060 Laptop GPU:

| Stage | Median (ms) | p95 (ms) |
|---|---|---|
| Query embedding (bi-encoder, GPU) | 6.5 | 10.2 |
| FAISS top-10 search | 0.06 | 0.11 |
| **Pipeline 1 retrieval** | **6.6** | **10.4** |
| Cross-encoder, 10 pairs (GPU) | 20.1 | 30.6 |
| **Pipeline 2 retrieval (GPU re-rank)** | **27.4** | **38.8** |
| Cross-encoder, 10 pairs (CPU) | 200.6 | 230.4 |
| **Pipeline 2 retrieval (CPU re-rank)** | **208.1** | **237.1** |
| LLM generation (both pipelines) | 3047 | 6417 |

End-to-end median: Pipeline 1 ≈ 3054 ms, Pipeline 2 ≈ 3075 ms (re-ranking ≈ 0.7% of total on GPU).
Figure: `artifacts/figures/latency_bars.png`.

YOUR WORDS: is the extra time worth it here? GPU vs CPU difference? What dominates end-to-end time?

## 4. Scaling question

Measured cost of the cross-encoder ≈ **1.7 ms per chunk (GPU)**, **17 ms per chunk (CPU)**, growing linearly (`ce_scaling.png`):

| Corpus size | CE on everything, GPU | CE on everything, CPU | Two-stage |
|---|---|---|---|
| 92 (this Act) | 0.16 s | 1.6 s | ~27 ms |
| 10,000 | 17 s | 2.8 min | ~27 ms + ANN |
| 1,000,000 | 29 min | 4.7 h | ~27 ms + ANN |
| 100,000,000 | 2.0 days | 19.7 days | ~27 ms + ANN |

On this tiny corpus, CE over all chunks *was* more accurate (Hit@3 0.917 vs 0.792), because the first stage missed 5 gold chunks.

YOUR WORDS. Points to cover: nothing can be pre-computed or indexed for a cross-encoder; cost grows linearly with corpus size and is paid on every query;
the bi-encoder's cost is mostly offline; the trade-off (bi-encoder for recall, cross-encoder for precision on a short list); how you would raise first-stage recall instead (larger k, hybrid BM25 + dense).
