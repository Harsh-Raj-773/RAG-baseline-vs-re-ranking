# RAG over the DPDP Act, 2023: Baseline vs. Cross-Encoder Re-Ranking

Two retrieval pipelines answer questions about one complex legal PDF, India's **Digital Personal Data Protection Act, 2023**
(official Gazette copy, 21 pages). Both use the same frozen chunk corpus.

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Retrieve | bi-encoder `all-MiniLM-L6-v2` + FAISS inner product (cosine), **top 10** | the **same** top 10 |
| Re-rank | none | cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs |
| Generate | top 3 → LLM → **Answer 1** | top 3 after re-ranking → LLM (identical prompt) → **Answer 2** |

LLM: **Qwen2.5-3B-Instruct**, run locally on the GPU with greedy decoding. No API key is needed.

Everything is in one notebook, executed with all outputs saved: **[`rag_baseline_vs_rerank.ipynb`](rag_baseline_vs_rerank.ipynb)**

## Results (24 queries)

| | Hit@1 | Hit@3 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| Pipeline 1 (baseline) | 0.458 | 0.667 | 0.589 | 0.792 |
| Pipeline 2 (re-ranked) | 0.708 | 0.792 | 0.743 | 0.792 |
| Cross-encoder over all 92 chunks (no first stage, for reference) | 0.750 | 0.917 | 0.833 | 0.958 |

* **3 queries fixed** by re-ranking (gold chunk moved into the top 3): K6 keyword trap #7→#1, E6 exception #4→#1, N3 negation #4→#1. **0 hurt.**
* **5 queries out of reach**: the gold chunk was not in the bi-encoder's top 10, so the cross-encoder could not recover it. One of them (B5) makes both pipelines hallucinate.
* Answer quality (manual labels): N3 went from refusal to correct, and K1 from a misstated penalty (Rs 200 crore) to the correct Rs 250 crore. In other cases a correct chunk in the context was not enough; the 3B model sometimes misread negations (E6, K6).

| Median latency per query | ms |
|---|---|
| Pipeline 1 retrieval (embed + FAISS) | 6.6 |
| Pipeline 2 retrieval, re-rank on GPU | 27.4 |
| Pipeline 2 retrieval, re-rank on CPU | 208.1 |
| LLM generation (same for both) | 3047 |

Cross-encoding every chunk costs about 1.7 ms per chunk on GPU (17 ms on CPU), or **about 29 minutes per query for 1 million chunks**.
The two-stage pipeline stays at about 27 ms because the first stage uses precomputed vectors.

## Notebook outline

1. Setup  2. Load the PDF (downloaded from MeitY, SHA-256 printed)  3. Structure-aware parsing and chunking
4. Pipeline 1  5. Pipeline 2  6. LLM and the identical prompt  7. Query set with gold labels
8. Retrieval evaluation on all queries  9. Failure-mode deep dives (rank shifts, bump charts, Answer 1 vs Answer 2)
10. Latency profiling  11. Scaling experiment  12. Saved artifacts

## How to run

Python 3.12, Windows/Linux, NVIDIA GPU recommended (a CPU works, just slower for the LLM).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace rag_baseline_vs_rerank.ipynb
```

* The PDF is in `data/`; if it is missing, the notebook downloads it.
* LLM answers are cached in `artifacts/llm_cache.json`, keyed by a prompt hash, so a re-run reproduces the same answers without regenerating them. Delete the file to regenerate.
* Model weights (about 6 GB for the LLM) download from Hugging Face on first run.

## Repository layout

```
rag_baseline_vs_rerank.ipynb   the notebook (main deliverable)
data/dpdp_act_2023.pdf         the source document
artifacts/chunks.jsonl         the frozen corpus (92 chunks)
artifacts/results/<qid>.json   per query: 10 candidates with both ranks/scores, both prompts, both answers
artifacts/*.csv                rank shifts, metrics, answer labels, latency, scaling projection
artifacts/figures/             rank-shift plot, bump charts, latency bars, scaling plot
report/                        the written report
```
