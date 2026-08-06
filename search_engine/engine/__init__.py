"""
engine
======

Responsibility
--------------
The public-facing FACADE for the entire search engine. This is the only
module a consumer of this library should need to import directly.

Design pattern in play: Facade
-----------------------------------
Everything below this module -- analysis, indexing, index structures,
query parsing, retrieval, ranking, storage -- is real complexity that a
caller should not need to understand just to index a document or run a
search. `SearchEngine` hides that complexity behind two operations:

    engine.index(document)
    engine.search(query_string, top_k=10)

Internally, `SearchEngine` wires the other modules together:
    index(doc):
        analysis.Analyzer -> indexing.Indexer -> index.Segment -> storage

    search(query_string):
        analysis.Analyzer (on the query text)
        -> query.parser -> query.QueryNode tree
        -> retrieval.executor -> candidate set
        -> ranking.Scorer -> scored, sorted top-K
        -> fetch stored fields for top-K only (NOT the whole candidate
           set -- fetching stored data for every candidate, rather than
           just the final top-K shown to the user, is wasted I/O that
           does not scale)

Planned contents (not yet implemented):
  - search_engine.py   : The SearchEngine facade class itself.
  - config.py           : Central configuration (which Analyzer, which
                          Scorer strategy, merge policy settings, etc.),
                          so swapping strategies is a config change, not
                          a code change, wherever reasonably possible.
"""
