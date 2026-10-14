# MEMORANDUM

**TO:** Leadership
**FROM:** Shridhar Pathak
**DATE:** October 13, 2026
**RE:** Proposal: a starting evaluation methodology for document AI quality

## Context

We are scaling document ingestion toward tens of thousands of documents,
parsed by machine and served through AI question-answering. At that scale,
nobody can read the output to check it. Quality becomes something we
measure, or something we simply hope for.

## The problem

A document pipeline can look healthy while being wrong in ways that
matter: a table parsed with shifted columns puts the wrong number in
front of a decision-maker; a Q&A system that never says "I don't know"
will hallucinate confidently instead. These failures are silent. Without
an evaluation methodology, we find out from users, not from our systems.

## The proposal

Adopt a four-layer evaluation methodology as our starting standard for
any document AI work:

1. **Parse fidelity.** Does the parsed output faithfully represent the
   source? Measured per document: text coverage against the source,
   table structure accuracy (rows x columns, values in the right cells),
   empty-chunk rate, header/footer leakage. Golden documents with known
   probe facts are asserted on every parser change.
2. **Retrieval quality.** Does the system find the right content?
   Measured as hit rate and mean reciprocal rank over a golden Q&A set
   built from our own corpus, sliced by document and query type.
3. **Answer quality.** Is the response correct and honest? Expected-fact
   checks for extractive answers; correctness, faithfulness, and
   abstention checks for generated answers. Unanswerable questions must
   refuse, not invent.
4. **Production monitoring.** Every query logged; abstention rate,
   zero-result rate, user downvote rate, and latency on a dashboard, with
   nightly re-runs of the golden set alerting on regression.

The full methodology, including the golden-set construction protocol
and sample-size guidance, is documented alongside the pilot code.

## What is already proven

I built a pilot pipeline this week: documents parsed through a
production document API, normalized into a searchable knowledge base,
with a web Q&A interface. Against a real 10-Q filing, the baseline
measures retrieval hit rate 1.00 with MRR 0.93, answer score 0.78, and
clean parse health (75 chunks, zero empty, 26 table-bearing). The
framework has already caught and fixed two real defects, including a
case where the system answered an unanswerable question instead of
abstaining.

## Cost

Parsing 10,000 documents (roughly 10 pages each) costs on the order of
$1,200 using a mixed accuracy/cost engine profile, plus roughly $50 per
month in infrastructure. The evaluation buildout itself is primarily
human time: about a day to establish the first golden set, then minutes
per week to maintain. Compared against the cost of acting on wrong
numbers, it is the cheapest insurance in the system.

## Ask

1. Approve this methodology as the starting standard for document AI
   quality on our team.
2. Next steps on approval: grow the golden set to decision-grade size
   (~50 cases), add nightly regression runs, and extend query logging
   to production traffic so the eval set grows out of real usage.
