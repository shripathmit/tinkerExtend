# Evaluation Methodology: Document AI Quality

How we know the knowledge base is good — and how we know a change made it
better or worse. This document is the standing methodology; the runnable
implementation lives in `src/tinker_extend/eval.py` and golden sets live
alongside it as versioned JSONL.

## Principles

1. **Eval is a measurement instrument, not a test suite.** Its job is to
   answer: "is the system good enough to trust?" and "did my last change
   make it better or worse?"
2. **Golden sets are corpus-specific.** Nobody sells verified Q&As about
   your documents. You build them from your own corpus, or you buy the
   labeling labor (Scale AI, Surge AI, Labelbox) — not the data.
3. **Never tune to the golden set.** It is a thermometer, not the target.
   Keep a held-out split you never look at during development.
4. **Slice, don't average.** A 0.95 average can hide a completely broken
   document category. Report every metric per doc type and query type.
5. **Version everything.** Golden sets live in git next to the code that is
   measured against them.

## The four layers

### Layer 1 — Parse fidelity ("did we read the document correctly?")

Per-document metrics, computed at ingest time, no ground truth needed:

| Metric | How | Red flag |
|---|---|---|
| text_coverage | parsed chars ÷ source PDF text-layer chars | < 0.95 on clean digital PDFs |
| table_recall | known tables found ÷ known tables (golden docs) | < 1.0 |
| table_shape_accuracy | tables with correct rows × cols | < 0.9 |
| empty_chunk_rate | chunks with < 20 chars ÷ total chunks | > 0.05 |
| garbled_rate | chunks matching mojibake patterns | > 0 |
| header_footer_leakage | repeated n-grams across pages inside body chunks | elevated vs baseline |
| parse failure rate | by failureReason | any spike |

Deeper checks use **golden documents**: 5–10 docs spanning your types
(clean digital, scanned, table-heavy, multi-column, footnote-dense), each
with 10–20 probe facts and known table dimensions recorded once, asserted
on every run. Re-run goldens on every parser version or config change —
this is the highest-ROI eval habit in the system.

### Layer 2 — Retrieval ("did we find the right chunks?")

Golden Q&A set with known-relevant content. Metrics:

- **hit_rate@k**: fraction of questions with ≥ 1 relevant chunk in top-k.
- **MRR** (mean reciprocal rank): 1 ÷ rank of the first relevant chunk,
  averaged. Rewards putting the right chunk first.

Report per slice (doc type × query type). Target on a clean corpus:
hit_rate@5 ≥ 0.90.

### Layer 3 — Answer quality ("is the response correct and honest?")

- **Extractive answers**: expected-fact presence checks — automatable,
  no judge needed.
- **Generated answers**: three checks —
  1. *correctness* vs the expected answer,
  2. *faithfulness*: every claim cited and actually supported (checkable
     with an NLI model or a second LLM call),
  3. *abstention*: unanswerable questions must refuse, not hallucinate.
- **LLM-as-judge** (optional): works, but calibrate it. Judges have
  position bias and prefer their own style. Spot-check judge scores
  against human grades on a sample, and keep the judge blind to which
  system produced the answer.

### Layer 4 — Production ("what are real users experiencing?")

Log every query: question, retrieved chunks, answer, latency, feedback.
Dashboard four numbers: **abstention rate** (a spike means retrieval broke
or a corpus gap), **zero-result rate**, **downvote rate**, **p99 latency**.
Re-run the golden set nightly and alert on regression. Downvoted answers
become new golden cases — the set grows out of real usage.

## Building a golden set

1. **Sample.** Stratify documents by type, length bucket, and
   parse-confidence bucket (low-confidence parses get *more* eval
   attention). Sample proportionally within strata, minimum ~10 docs per
   stratum so small strata aren't noise. Then sample 2–3 diverse chunks
   per doc (a table chunk, a prose chunk, a random one) — never chunks
   directly, or three long documents dominate the set.
2. **Draft with an LLM.** Generate candidate (question, answer,
   supporting chunk) triples from sampled chunks. Fast and cheap.
3. **Verify by human.** Non-negotiable. 2–3 minutes per pair: is the
   question natural, is the answer exactly right, does the cited chunk
   support it? Unverified LLM drafts are the system's own beliefs
   reflected back at itself.
4. **Add adversaries by hand.** Unanswerables, paraphrases, typos,
   lookalike traps. Highest value per minute spent.
5. **Version it.** JSONL in git, next to the pipeline. Format:
   `{"question", "relevant_contains", "answer_contains", "answerable"}`.

## Sample-size guidance

To detect a Δ-point change in hit_rate at 95% confidence, you need
roughly n ≈ (1.96² × 0.25) / Δ² cases (worst case). In practice:

- ~10 cases: smoke signal. One miss swings the score 10+ points.
- ~50 cases: decision-grade for large regressions (what our pilot uses).
- ~200–400 cases: trustworthy for 5–10 point changes. The sweet spot
  at 10k+ documents.
- Hold out 20% that you never inspect during development; use only for
  release decisions.

## Maintenance

- Re-run the full eval on every pipeline or parser change; treat a
  hit_rate or abstention dip as a build failure.
- Resample from new arrivals quarterly — golden sets rot as the corpus
  drifts.
- Re-run goldens whenever the parse provider ships a new API version.

## Running it here

```bash
python -m tinker_extend.cli --db ./kb.sqlite3 eval \
  --eval-file eval/golden_10q.jsonl --k 5
```

Current baseline (Moderna 10-Q pilot, 9 cases): retrieval hit_rate@5
1.00 / MRR 0.93; answer mean score 0.78; parse health clean (75 chunks,
0 empty, 26 with tables).
