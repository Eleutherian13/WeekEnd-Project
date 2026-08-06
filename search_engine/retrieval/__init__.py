"""
retrieval
=========

Responsibility
--------------
Executes a `query.QueryNode` tree against `index` structures (Dictionary,
PostingsList, PositionalPostingsList) to produce a CANDIDATE SET of
matching document ids.

This is pure set/boolean logic: "which documents satisfy this query
structurally?" It answers yes/no per document, not "how relevant is it."
Relevance scoring is `ranking`'s job, deliberately kept separate -- see
the module-level note in `ranking` for why.

Planned contents (not yet implemented):
  - executor.py        : Walks a QueryNode tree recursively, dispatching
                         to the right merge algorithm per node type.
  - boolean_merge.py    : AND (intersection) / OR (union) / NOT
                         (difference) algorithms over sorted postings
                         lists. Sorted-order is what makes these O(n+m)
                         merges instead of O(n*m) nested scans -- the same
                         reason PostingsList enforces sorted-by-doc_id.
  - phrase_matcher.py   : Given two or more terms' position lists for the
                         same document, checks whether they appear as
                         consecutive positions (an exact phrase match).

Connection to later milestones
--------------------------------
Skip lists (for faster AND-merges by letting one postings pointer "jump"
ahead) and lazy/iterator-based postings traversal (rather than fully
materializing lists in memory) both live here later -- once the basic
merge algorithms are correct and tested first.
"""
