"""
ranking
=======

Responsibility
--------------
Scores a CANDIDATE SET of documents (produced by `retrieval`) by
estimated relevance to the query, and sorts them.

Why this is separate from `retrieval`
----------------------------------------
Retrieval is exact, structural, set-based ("does this doc match at all?").
Ranking is statistical and swappable ("how relevant is it, on a
continuous scale?"). Keeping them apart means:
  1. Retrieval stays simple, fast, and correct without ever touching
     scoring math.
  2. Ranking functions can be swapped (TF-IDF -> BM25 -> a future ML
     re-ranker) via the Strategy pattern below, without retrieval logic
     changing AT ALL.
  3. It enables two-stage retrieval later: cheap retrieval over millions
     of docs -> cheap ranking to get top-K candidates -> expensive
     ranking (e.g. a neural re-ranker in the RAG milestone) applied only
     to that small top-K. This pattern is standard in production search
     and recommendation systems and only works because the stages are
     decoupled.

Design pattern in play: Strategy
------------------------------------
All scorers implement a common interface: score(query, doc_id, index)
-> float. `ranking` (or `engine`) picks which strategy to use at query
time without the caller needing to know the scoring math.

Planned contents (not yet implemented):
  - scorer.py       : Abstract base Scorer interface.
  - tfidf_scorer.py  : TF-IDF scoring strategy.
  - bm25_scorer.py    : BM25 scoring strategy (term saturation + document
                       length normalization -- improves on raw TF-IDF).
  - top_k.py          : Efficient top-K selection (heap-based) rather than
                       sorting the entire candidate set when K is small
                       relative to the candidate count.
"""
