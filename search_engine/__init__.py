"""
search_engine
=============

A from-scratch, production-style search engine built to understand the
architecture behind real-world systems like Apache Lucene and Elasticsearch.

This is NOT a toy inverted index in a single file. It is organized into
modules that mirror how real IR systems separate concerns, because those
boundaries exist for real engineering reasons (concurrency, swappability,
testability) -- not just aesthetics.

Module map
----------
analysis   -- Text processing: tokenization, normalization, stopwords, stemming.
              Shared by BOTH indexing and querying so text is treated identically
              in both places (a query for "Running" must match a document
              analyzed to contain "run").

index      -- Core, passive data structures: Dictionary (term -> postings
              pointer), PostingsList, PositionalPostingsList, MetadataIndex,
              Segment. These classes know how to maintain their own
              invariants (e.g. postings stay sorted by doc_id) but do not
              know how documents get *into* them.

indexing   -- The offline pipeline: raw documents -> analysis -> index
              data structures -> persisted segment. Orchestration lives
              here; the data structures it builds live in `index`.

query      -- Parses a query string / query spec into a structured Query
              object tree (Composite pattern: TermQuery, PhraseQuery,
              BooleanQuery). Knows nothing about *how* to execute a query
              against an index -- only how to represent one.

retrieval  -- Executes Query objects against index structures to produce a
              CANDIDATE SET of matching documents (boolean/set logic:
              postings intersection, union, phrase position matching).
              Deliberately knows nothing about relevance scoring.

ranking    -- Scores candidate documents for relevance (TF-IDF, BM25, and
              later, pluggable/ML rankers) using the Strategy pattern.
              Deliberately knows nothing about how candidates were found.

storage    -- Disk persistence for segments: serialization format,
              compression, and segment merge policy. Infrastructure
              concern, orthogonal to what an index *means* logically.

engine     -- The public-facing Facade. A SearchEngine class exposing
              index(doc) and search(query) as the only two things a
              consumer needs to know about.

Design principle used throughout
---------------------------------
Every module boundary above exists because the two sides of that boundary
change for different reasons, at different times, for different people.
That is the real definition of "separation of concerns" -- not just
"smaller files."
"""
