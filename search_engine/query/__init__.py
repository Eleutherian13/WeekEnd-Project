"""
query
=====

Responsibility
--------------
Represents a search query as a structured OBJECT TREE, and parses raw
query input (a string, or a structured query spec) into that tree.

This module knows how to REPRESENT a query. It deliberately does NOT know
how to EXECUTE one against an index -- that is `retrieval`'s job. This
split means adding a new query type (e.g. wildcard, fuzzy) is a two-step,
localized change: add the node type here, add its execution logic in
`retrieval`, and nothing else in the system needs to change.

Design pattern in play: Composite
------------------------------------
A boolean query like `AND(quick, OR(brown, fox))` is naturally a tree.
The Composite pattern lets leaf nodes (a single term) and composite nodes
(AND/OR combinations) share a common interface, so `retrieval` can
execute any node in the tree the same way, recursively, without knowing
whether it's a leaf or a branch.

Planned contents (not yet implemented):
  - query_node.py    : Abstract base QueryNode -- the common interface
                       all query types implement.
  - term_query.py     : TermQuery -- a single term match (leaf node).
  - phrase_query.py   : PhraseQuery -- an ordered sequence of terms that
                       must appear as consecutive positions (leaf node,
                       but requires a PositionalPostingsList to execute).
  - boolean_query.py  : BooleanQuery -- AND/OR/NOT combinations of other
                       QueryNodes (composite node).
  - parser.py         : Parses a raw query string (or structured spec)
                       into a QueryNode tree. This is where query text
                       gets passed through `analysis` (the same
                       tokenizer/normalizer used at index time).
"""
