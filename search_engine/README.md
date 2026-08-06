# search-engine

A from-scratch, production-style search engine, built to understand the
architecture behind Apache Lucene and Elasticsearch — not to be a toy
inverted index in a single file.

## Status

**Skeleton stage.** Every module below exists as a package with a
docstring describing its contract and responsibility. No implementation
logic has been written yet — that happens milestone by milestone.

## Architecture

```
search_engine/
├── analysis/    # Tokenization, normalization, stopwords, stemming.
│                # Shared identically by indexing AND querying.
├── index/       # Passive data structures: Dictionary, PostingsList,
│                # PositionalPostingsList, MetadataIndex, Segment.
├── indexing/    # Offline pipeline: raw docs -> analysis -> index structures.
├── query/       # Parses query strings into a QueryNode tree (Composite).
├── retrieval/   # Executes QueryNode trees -> candidate document set.
│                # Pure set logic. Knows nothing about relevance scoring.
├── ranking/     # Scores candidates for relevance (TF-IDF, BM25, ...).
│                # Strategy pattern. Knows nothing about how candidates
│                # were found.
├── storage/     # Segment persistence, compression, merge policy.
│                # Segments are immutable once written (LSM-tree-like).
└── engine/      # Public Facade: SearchEngine.index() / .search().
```

See each module's `__init__.py` docstring for the detailed rationale
behind its boundaries — the module split is deliberate, not decorative.

## Why indexing and querying are separate pipelines

Indexing is write-heavy and latency-tolerant. Querying is read-heavy and
latency-critical. They optimize for opposite things, so they never share
a code path or a mutable data structure.

## Why segments are immutable

Once written, a segment is never modified. New documents create new
segments; deletes are tombstone markers, not rewrites. This makes
concurrent reads lock-free and crash-safe, at the cost of periodic
background merge work — the same tradeoff LSM-tree storage engines
(RocksDB, Cassandra) make.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

1. Analysis pipeline (tokenizer -> normalizer -> stopwords -> stemmer)
2. Core index data structures (Dictionary, PostingsList)
3. In-memory indexing pipeline
4. Boolean retrieval (AND/OR/NOT merges over postings)
5. Positional index + phrase queries
6. TF-IDF and BM25 ranking
7. Segment persistence + compression
8. Segment merging
9. Fuzzy search, autocomplete (FST), hybrid/vector search, distributed search
