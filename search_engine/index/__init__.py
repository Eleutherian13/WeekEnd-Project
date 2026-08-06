"""
index
=====

Responsibility
--------------
Defines the PASSIVE core data structures of the search engine: the things
that ARE the index, as opposed to the code that builds them (`indexing`),
executes queries against them (`retrieval`), or persists them (`storage`).

These are implemented as CLASSES, not plain dicts, because they have:
  - behavior (e.g. PostingsList.add() must preserve sorted-by-doc_id order)
  - invariants to protect (breaking sort order silently breaks every
    downstream merge algorithm in `retrieval`)
  - multiple planned implementations later (e.g. Dictionary backed by a
    hash map now, later by a Finite State Transducer for prefix queries)

Planned contents (not yet implemented):
  - dictionary.py         : Dictionary class. term -> postings pointer +
                             document frequency. The in-memory "table of
                             contents" for the index.
  - postings.py            : PostingsList class. doc_id -> term_frequency,
                             kept sorted by doc_id to enable O(n+m) merges.
  - positional_postings.py : PositionalPostingsList, extending PostingsList
                             with per-occurrence position lists, required
                             for phrase queries.
  - metadata_index.py      : MetadataIndex class. Per-document and
                             corpus-level statistics (doc length, total doc
                             count, average doc length) needed by ranking
                             functions like BM25 -- NOT part of the term
                             postings, because it has a different access
                             pattern (looked up per candidate doc during
                             scoring, not iterated like a postings list).
  - segment.py              : Segment class. A self-contained, IMMUTABLE
                             bundle of {Dictionary, PostingsLists,
                             MetadataIndex, stored fields} representing one
                             slice of the index. Immutability here is what
                             will make concurrent reads lock-free later.

Why this module has NO logic for "how documents get indexed"
--------------------------------------------------------------
That is deliberately `indexing`'s job. This module only defines what a
valid index structure IS and how to safely mutate it. Keeping
"data + invariants" separate from "orchestration" means `retrieval` and
`ranking` can depend on `index` alone, without pulling in the entire
indexing pipeline.
"""
