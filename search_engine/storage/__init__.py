"""
storage
=======

Responsibility
--------------
Persists `index.Segment` objects to disk, and manages the lifecycle of
segments over time: writing new segments, reading them back, and merging
smaller segments into larger ones in the background.

Why immutable segments (recap -- see architecture discussion)
------------------------------------------------------------------
Index segments, once written, are NEVER modified in place. New documents
create new segments. Deletes are recorded as a "tombstone" marker on a
doc_id, not by rewriting the segment. This is the same core idea as an
LSM-tree (used by RocksDB, Cassandra, LevelDB): immutability makes
concurrent reads lock-free (a query just sees "the segments that existed
when the search started"), crash-safe (a failed write can't corrupt an
existing complete segment), and OS-cache-friendly (immutable data can be
memory-mapped and cached aggressively).

The tradeoff is write amplification: the same logical document's data
gets rewritten multiple times as segments merge over their lifetime.
This tradeoff, and the policy that decides *when* to merge, is a real
tuning knob we will implement and reason about explicitly, not hand-wave.

Planned contents (not yet implemented):
  - segment_writer.py   : Serializes an index.Segment to disk.
  - segment_reader.py    : Deserializes a segment from disk back into
                          queryable in-memory (or memory-mapped) form.
  - codec.py              : Encoding/compression of postings lists
                          (variable-byte encoding, delta encoding of
                          doc_id gaps, etc.) -- this is what keeps
                          postings lists small enough to be I/O and
                          cache efficient at scale.
  - merge_policy.py       : Decides when and which segments to merge,
                          balancing query-time cost (too many small
                          segments = more lists to merge per query)
                          against merge cost (too-aggressive merging =
                          wasted rewrite work).
"""
