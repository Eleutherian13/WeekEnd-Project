"""
indexing
========

Responsibility
--------------
The OFFLINE pipeline that turns raw documents into the data structures
defined in `index`. This is the "write path" of the search engine.

Offline vs online (why this is its own module)
-------------------------------------------------
Indexing is write-heavy, batch-oriented, and latency-TOLERANT (a document
can take 500ms to become searchable; nobody is blocked waiting on it).
Querying (see `retrieval`) is read-heavy and latency-CRITICAL
(sub-100ms expectations). These two paths optimize for opposite things,
so they must not share code paths or data-structure choices -- a structure
tuned for fast writes is rarely also optimal for fast concurrent reads.

Planned contents (not yet implemented):
  - document.py        : Document class -- the raw input unit (doc_id,
                         fields such as title/body, raw text).
  - indexer.py          : Indexer class. Orchestrates:
                             raw Document
                             -> analysis.Analyzer (tokenize/normalize)
                             -> build in-memory index.* structures
                             -> hand off to storage for persistence
  - segment_builder.py  : Builder-pattern object that accumulates terms
                         and postings incrementally, then "seals" them
                         into an immutable index.Segment. Builder pattern
                         fits here because a Segment is complex and
                         immutable once finished -- you cannot construct
                         it in one atomic step from raw input.

Connection to later milestones
--------------------------------
This is where SPIMI (Single-Pass In-Memory Indexing) and external-merge
construction will eventually live, once corpus size exceeds available RAM.
For now we start with a simple in-memory build to master the data
structures first -- premature scaling here would be optimizing before we
understand what we're scaling.
"""
