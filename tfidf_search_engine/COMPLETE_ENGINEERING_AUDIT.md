# tfidf_search_engine - Complete Engineering Audit

Audit basis: source inspection and runtime checks performed on 2026-09-16. This document describes the code that exists; it does not define a desired architecture. No source files were modified as part of this audit.

## 1. Project Map

### Analysis

| File                          | Purpose                                    | Key API                                      | Dependencies / callers                                                                                  | Status                                                |
| ----------------------------- | ------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `analysis/analyzer.py`        | Default text-analysis pipeline             | `Analyzer.analyze(str) -> list[str]`         | Regex tokenizer, normalizer, stopwords, stemmer/lemmatizer; used by `IndexBuilder` and `QueryProcessor` | IMPLEMENTED                                           |
| `analysis/tokenizer.py`       | Tokenizer base plus an older `Tokenizerv1` | `Tokenizer.tokenize`, `Tokenizerv1.tokenize` | Regex and whitespace tokenizers                                                                         | DUPLICATED / PARTIAL                                  |
| `analysis/regex_tokenizer.py` | Regex word tokenization                    | `RegexTokenizer.tokenize`                    | `Analyzer`                                                                                              | IMPLEMENTED                                           |
| `analysis/normalizer.py`      | Lowercases tokens                          | `Normalizer.normalize`                       | `Analyzer`                                                                                              | IMPLEMENTED                                           |
| `analysis/stopwords.py`       | Removes fixed stopword set                 | `StopWordRemover.remove`                     | `Analyzer`                                                                                              | IMPLEMENTED                                           |
| `analysis/stemmer.py`         | NLTK Porter stemming                       | `Stemmer.stem`                               | `Analyzer`, `ForwardIndex`, `Statistics`                                                                | IMPLEMENTED; dependency unverified                    |
| `analysis/lemmetization.py`   | NLTK WordNet lemmatization                 | `Lemmatizer.lemmatize`                       | `Analyzer` when stemming is disabled                                                                    | IMPLEMENTED; misspelled filename; resource unverified |
| `analysis/whitespace.py`      | Whitespace tokenization                    | `WhitespaceTokenizer.tokenize`               | No discovered caller                                                                                    | IMPLEMENTED / UNUSED                                  |
| `analysis/spacy_tokenizer.py` | spaCy tokenization                         | `SpacyTokenizer.tokenize`                    | No discovered caller; loads model at import                                                             | PARTIAL / UNVERIFIED                                  |

### Documents

| File                    | Purpose                       | Key API                                           | Status                                                                                    |
| ----------------------- | ----------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `document/documents.py` | Validated document data model | `Document(document_id: int, text: str, metadata)` | IMPLEMENTED                                                                               |
| `document/corpus.py`    | In-memory document collection | `Corpus.add/get/remove`, iteration, membership    | PARTIAL; `__setitem__` ignores its key and missing-item behavior differs from annotations |
| `document/chunking.py`  | Validated chunk data model    | `Chunk(chunk_id, text, metadata, position)`       | IMPLEMENTED / UNUSED; no chunker exists                                                   |

### Index

| File                        | Purpose                                        | Key API                                           | Status                                                                             |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `index/builder.py`          | Builds lexical indexes from analyzed documents | `IndexBuilder.add_document`, `build`              | IMPLEMENTED; in-memory only                                                        |
| `index/forward_index.py`    | Document-to-term-frequency map                 | `ForwardIndex.add_document`, term lookup, lengths | IMPLEMENTED                                                                        |
| `index/inverted_index.py`   | Term-to-posting-list map                       | `add`, `get`, membership                          | IMPLEMENTED                                                                        |
| `index/posting.py`          | One term/document posting                      | `Posting(document_id, term_frequency)`            | IMPLEMENTED                                                                        |
| `index/posting_list.py`     | Unique postings for one term                   | `add`, `get`, iteration                           | IMPLEMENTED; does not enforce sorted insertion                                     |
| `index/positional_index.py` | Term/document positions                        | `add`, `get_positions`, `get_documents`           | IMPLEMENTED                                                                        |
| `index/vocabulary.py`       | Term-to-integer ID map                         | `add`, `contains`, `get_id`                       | PARTIAL; `get_id` annotation suggests optional result but raises for unknown terms |
| `index/statistics.py`       | Corpus statistics                              | counts, lengths, document frequency               | IMPLEMENTED; its example uses the wrong `IndexBuilder` constructor                 |
| `index/index_builder.py`    | Duplicate-looking builder module               | None                                              | EMPTY / UNUSED                                                                     |

### Query

| File                        | Purpose                           | Status                             |
| --------------------------- | --------------------------------- | ---------------------------------- |
| `query/query.py`            | Validated raw query object        | IMPLEMENTED                        |
| `query/query_parser.py`     | Wraps raw text in `Query`         | IMPLEMENTED / UNUSED by the facade |
| `query/query_processing.py` | Analyzes `Query` into `list[str]` | IMPLEMENTED                        |
| `query/boolean_query.py`    | Term, AND, OR, NOT evaluation     | IMPLEMENTED / DISCONNECTED         |
| `query/phrase_query.py`     | Positional phrase matching        | IMPLEMENTED / DISCONNECTED         |
| `query/fuzzy_query.py`      | Levenshtein vocabulary matching   | IMPLEMENTED / DISCONNECTED         |
| `query/prefix_query.py`     | Prefix vocabulary matching        | IMPLEMENTED / DISCONNECTED         |
| `query/wildcard_query.py`   | `*` and `?` wildcard matching     | IMPLEMENTED / DISCONNECTED         |
| `query/synonym_query.py`    | Configured synonym lookup         | IMPLEMENTED / DISCONNECTED         |
| `query/query_expansion.py`  | User-provided term expansion      | IMPLEMENTED / UNUSED               |

### Ranking

| File                  | Purpose                          | Status                                          |
| --------------------- | -------------------------------- | ----------------------------------------------- |
| `ranking/bm25.py`     | BM25 term and document scoring   | IMPLEMENTED standalone                          |
| `ranking/tfidf.py`    | TF-IDF term/document scoring     | IMPLEMENTED standalone                          |
| `ranking/cosine.py`   | List-based cosine implementation | DUPLICATED; vector code uses `VectorSimilarity` |
| `ranking/fusion.py`   | Weighted numeric score fusion    | IMPLEMENTED standalone; no caller found         |
| `ranking/reranker.py` | Sorts integer-ID score maps      | IMPLEMENTED standalone; no caller found         |

### Retrieval and search

| File                                 | Purpose                                       | Status                                         |
| ------------------------------------ | --------------------------------------------- | ---------------------------------------------- |
| `retieval/candidate_generator.py`    | OR-union candidate generation                 | IMPLEMENTED and manually exercised             |
| `retieval/lexical/retriever.py`      | Abstract lexical retriever contract           | PARTIAL / BROKEN import                        |
| `retieval/lexical/bm25_retriever.py` | Candidate generation plus BM25 ranking        | BROKEN import and scorer call                  |
| `retieval/dense/retriever.py`        | Raw text dense retrieval via `VectorStore`    | IMPLEMENTED in isolation                       |
| `retieval/dense/vector_retriever.py` | Thin `DenseRetriever` subclass                | DUPLICATED / BROKEN import                     |
| `retieval/graph/retriever.py`        | Abstract graph retrieval contract             | BROKEN import; missing `RetrievalResult`       |
| `retieval/graph/graph_retriever.py`  | BFS graph retrieval from supplied start nodes | BROKEN import; query does not affect traversal |
| `search/search_engine.py`            | Intended lexical search facade                | BROKEN; calls undefined methods                |

The package directory is named `retieval`, but several modules import `retrieval`.

### Vector

| File                                        | Purpose                                            | Status                                                |
| ------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- |
| `vector/embedding.py`                       | Abstract embedding interface and validation        | IMPLEMENTED                                           |
| `vector/embeddings/ollama.py`               | Ollama provider                                    | IMPLEMENTED source / runtime unverified               |
| `vector/embeddings/openai.py`               | OpenAI provider                                    | IMPLEMENTED source / runtime unverified; not exported |
| `vector/embeddings/sentence_transformer.py` | SentenceTransformers provider                      | IMPLEMENTED source / runtime unverified               |
| `vector/similarity.py`                      | Dot, cosine, Euclidean operations                  | IMPLEMENTED and used                                  |
| `vector/vector_store.py`                    | In-memory vector/metadata storage and exact search | IMPLEMENTED / TESTED                                  |
| `vector/hnsw.py`                            | Separate approximate nearest-neighbor index        | IMPLEMENTED / TESTED separately                       |
| `vector/persisence.py`                      | JSON persistence for store and optional HNSW       | IMPLEMENTED source / misspelled / UNTESTED            |

### Graph, hybrid, evaluation, and other files

| Area         | Files                                                                              | Status                                                                                         |
| ------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Graph        | `graph/node.py`, `edge.py`, `adjacency.py`, `graph.py`, `traversal.py`, `store.py` | BROKEN as a subsystem; field names, imports, and method names disagree                         |
| Hybrid       | `hybrid/fusion.py`                                                                 | STUB; abstract method ends in `NotImplementedError` and imports missing result type            |
| Evaluation   | `evaluations/*.py`                                                                 | Metric implementations exist, but package initializer imports nonexistent/mismatched filenames |
| Examples     | `app.py`, `booleanSearch.py`, module `__main__` blocks                             | MANUAL / PARTIAL; not an integrated application                                                |
| Data         | `data/documents.json`                                                              | EMPTY / UNUSED in discovered code                                                              |
| Planning/RAG | `planning/`, `rag/`                                                                | EMPTY; no implementation established                                                           |
| Utilities    | `utils/serialization.py`                                                           | EMPTY / UNUSED                                                                                 |
| Dependencies | `requirements.txt`                                                                 | EMPTY despite external imports                                                                 |

## 2. Subsystems

The project is a collection of source-level subsystems rather than one verified application:

1. Lexical analysis, indexing, candidate generation, and standalone rankers.
2. Exact vector storage and separate HNSW search.
3. Query operators that are not integrated into the main search facade.
4. Graph structures that currently cannot run.
5. Standalone evaluation metrics with broken package exports.
6. Empty planning, RAG, and serialization areas.

## 3. Architecture Reconstruction

### Lexical indexing

```text
Document or (document_id, text)
  -> Analyzer
  -> ForwardIndex
  -> PositionalIndex
  -> InvertedIndex + PostingList
  -> Vocabulary
  -> Statistics
```

`IndexBuilder` performs this flow in memory. It does not persist the index and does not retain a `Corpus`.

### Query and candidate generation

```text
raw query
  -> Query
  -> QueryProcessor
  -> list[str]
  -> CandidateGenerator.generate
  -> set[int]
```

`CandidateGenerator` has OR semantics only.

### Intended lexical ranking flow

```text
processed terms
  -> CandidateGenerator
  -> BM25Retriever
  -> BM25
  -> list[(int, float)]
```

This is not runnable as written. The lexical retriever imports `retrieval.candidate_generator`, which does not exist, and `BM25Retriever` calls `BM25.score` with the wrong argument shape/order.

### Actual search facade flow

```text
SearchEngine.search(text, mode)
  -> Query
  -> QueryProcessor
  -> CandidateGenerator.and_retrieve / or_retrive
```

Both final candidate-generator methods are absent. The facade never invokes BM25, TF-IDF, or a reranker.

### Dense flow

```text
raw query text
  -> DenseRetriever
  -> VectorStore.search_text
  -> EmbeddingModel.embed
  -> VectorStore.search
  -> exact similarity ranking
  -> list[(str, float)]
```

This path is source-level coherent when a configured embedding model is available.

### ANN flow

```text
VectorStore + separately inserted vectors
  -> HNSW graph
  -> HNSW.search
  -> list[HNSWSearchResult]
```

HNSW is not called by `VectorStore.search` or `DenseRetriever`.

### Graph flow

```text
caller-supplied start node IDs
  -> GraphTraversal.bfs
  -> document-node filtering
  -> inverse-hop score
  -> RetrievalResult objects
```

The graph flow cannot currently import or execute.

### Evaluation flow

```text
retrieved list[str] + relevant set[str]
  -> metric functions
  -> EvaluationResult
```

The metric source files exist, but the public package import fails.

## 4. Data Flow and Contracts

| Boundary         | Actual representation                                 |
| ---------------- | ----------------------------------------------------- |
| Document         | `Document(document_id: int, text: str, metadata)`     |
| Analyzed text    | `list[str]`                                           |
| Forward index    | `dict[int, dict[str, int]]`                           |
| Inverted index   | `dict[str, PostingList]`                              |
| Posting          | integer document ID and integer term frequency        |
| Positional index | term -> document ID -> positions                      |
| Candidates       | `set[int]`                                            |
| Lexical results  | `list[tuple[int, float]]`                             |
| Vector results   | `VectorSearchResult(id: str, score: float, metadata)` |
| Dense results    | `list[tuple[str, float]]`                             |
| HNSW results     | `HNSWSearchResult(id: str, score: float)`             |
| Score fusion     | `dict[str, dict[int, float]] -> dict[int, float]`     |
| Reranking        | `dict[int, float] -> list[tuple[int, float]]`         |
| Evaluation       | `list[str]`, `set[str]`, optional `dict[str, float]`  |

The major incompatible boundary is identifier type: lexical and reranking code use `int`, while vector, graph, and evaluation contracts use `str`. No adapter or conversion policy exists.

Other contract mismatches:

- `BM25Retriever` calls `BM25.score(document_id, terms)`, but the implemented method is `score(term, document_id)`. `score_document(terms, document_id)` is available but unused.
- `SearchEngine` expects `and_retrieve` and misspelled `or_retrive`; `CandidateGenerator` exposes only `generate` and `__call__`.
- `CandidateGenerator` returns an unordered set; lexical retrieval promises ordered scored tuples.
- Graph and hybrid modules expect an undefined `RetrievalResult`.
- `VectorStore` and HNSW return result objects; dense retrieval converts them to tuples.

## 5. Lexical Retrieval Audit

### Analyzer to indexes

`Analyzer` defaults to regex tokenization, lowercase normalization, fixed stopword removal, and Porter stemming. `IndexBuilder.add_document` analyzes the text, computes term frequencies, writes positions, adds vocabulary terms, and creates postings.

The lexical structures are implemented but have limited formal coverage. Indexing is in-memory only.

### CandidateGenerator

`CandidateGenerator.generate(iterable[str]) -> set[int]`:

- Rejects `None`.
- Rejects non-string terms.
- Strips whitespace.
- Skips empty terms.
- Unions postings for known terms.
- Returns OR candidates.

It does not score, rank, provide AND semantics, or perform top-k selection.

### BM25 and TF-IDF

`BM25` and `TFIDF` are concrete standalone scorers backed by `ForwardIndex` and `Statistics`. They are not connected to `SearchEngine`.

`BM25Retriever` is the intended connection point but is currently blocked by both import and API mismatches.

### Specialized queries

Boolean, phrase, fuzzy, prefix, wildcard, synonym, and expansion classes directly inspect indexes or vocabulary. No discovered caller connects them to `QueryParser`, `QueryProcessor`, `SearchEngine`, or the rankers.

## 6. Dense / Vector Audit

`EmbeddingModel` owns text-to-vector conversion. `VectorStore` owns vector IDs, vectors, metadata, dimensions, and exact search. `DenseRetriever` validates raw query text and delegates to `VectorStore.search_text`.

`VectorStore.search` compares every stored vector. Cosine and dot product sort descending; Euclidean sorts ascending. Metadata filtering is exact key/value matching.

`HNSW` owns graph layers and ANN traversal while reading vectors from `VectorStore`. It does not own metadata and is not automatically synchronized with store mutations. Its tests populate the store and HNSW separately.

`Persistence` serializes private VectorStore and HNSW state to JSON and can reconstruct both, but there are no tests or discovered callers. Its filename is `persisence.py`.

Provider status is UNVERIFIED because `requirements.txt` is empty and providers require external packages/services/configuration.

## 7. Graph / Hybrid / Reranking Audit

The graph subsystem is BROKEN before retrieval can be reached:

- `Edge` uses `default_facotory` instead of `default_factory`.
- `AdjacencyList` uses `defaultdict` without importing it.
- `Graph` defines `add_nodes`, while callers use `add_node`.
- `GraphStore` calls missing `Graph.add_node` and `Graph.get_node`.
- `Graph` expects `edge.source`/`edge.target`, while `Edge` defines `source_node`/`target_node`.
- `Graph` uses both `adjacency` and `_adjacency`.
- Several required typing imports and adjacency methods are absent.
- Graph retrieval imports the nonexistent `retrieval` package and missing `RetrievalResult`.

`ranking.ScoreFusion` is a standalone integer-ID score-map combiner. `hybrid.FusionStrategy` is a separate abstract ranked-list contract and is a stub. `Reranker` is a standalone score sorter. No caller connects these components.

## 8. Search Facade Audit

`SearchEngine` is BROKEN and incomplete. It accepts a `QueryProcessor` and `CandidateGenerator`, constructs a `Query`, analyzes it, then calls nonexistent candidate-generator methods. It has no scorer, no ranking, no top-k behavior, and no dense/graph/hybrid path.

## 9. Testing and Verification Audit

### Passing verification

Command:

```text
python -m unittest test_vector_system.py -v
```

Observed result: 24 tests passed. Coverage includes VectorStore CRUD, duplicate IDs, dimensions, batch atomicity, cosine/dot/Euclidean search, metadata filtering, HNSW insertion, graph invariants, removal, clearing, nearest-neighbor checks, and random-data recall.

### Failing verification

`python app.py` fails before its manual assertions because `retieval.lexical.retriever` imports nonexistent `retrieval.candidate_generator`.

`python -m unittest discover -v` runs the vector tests but fails importing `evaluations` because `evaluations/__init__.py` imports nonexistent `evaluator.py`.

`import graph.graph` fails because `Edge` uses the invalid `default_facotory` keyword.

A direct BM25 retriever construction fails at the same `retrieval` import before reaching the scorer call mismatch.

### Partial or weak tests

- `tests/test_postings.py` contains assertions but invokes one test function directly.
- `tests/test_posting_list.py` and `tests/test_inverted_index.py` primarily print values and contain no meaningful assertions.
- `app.py` manually checks CandidateGenerator and a temporary `LexicalRetriever` implementation but does not test BM25.

### Untested or unknown

Analyzer, document models, index builder, statistics, query classes, BM25, TF-IDF, BM25Retriever, DenseRetriever, providers, persistence, graph, hybrid fusion, ScoreFusion, Reranker, SearchEngine, and Evaluator lack verified formal tests.

Compilation is not equivalent to runtime validation. The source compiles, but multiple imports and execution paths fail.

## 10. Blockers and Integration Gaps

### Blockers

1. `retieval` versus `retrieval` package mismatch prevents retrieval modules from importing.
2. `BM25Retriever` does not match the existing BM25 API.
3. `SearchEngine` calls undefined candidate-generator methods and does not rank.
4. Evaluation package exports reference absent filenames.
5. Graph classes disagree on fields, methods, and internal storage names.
6. Graph and hybrid code depend on an undefined `RetrievalResult`.

### Integration gaps

- No verified raw-text-to-ranked lexical path.
- No connection from SearchEngine to BM25 or TF-IDF.
- No lexical/dense fusion orchestrator.
- No HNSW use in the dense query path.
- No ScoreFusion or Reranker caller.
- No query-to-graph-node matching.
- No evaluation runner connected to retrieval output.
- No corpus loading from `data/documents.json`.
- No RAG implementation.

### Duplication

- `VectorRetriever` delegates unchanged to `DenseRetriever`.
- `ranking/cosine.py` overlaps with `vector/similarity.py`.
- `index/index_builder.py` duplicates the name of the implemented builder but is empty.
- `Tokenizerv1` overlaps with the abstract tokenizer design.

### Later features

HNSW integration, persistence round-trip testing, hybrid fusion, graph retrieval, query expansion, RAG, and scale benchmarking should follow a verified core retrieval path.

## 11. Current Development Frontier

```text
LAST WORKING COMPONENT
  VectorStore exact search and standalone HNSW search

CURRENTLY PARTIALLY WORKING COMPONENT
  IndexBuilder + CandidateGenerator + standalone BM25/TF-IDF

NEXT COMPONENT TO IMPLEMENT
  Importable and executable lexical retriever boundary

NEXT INTEGRATION TEST
  Build a small index, process a query, run BM25 retrieval, assert ranked IDs/scores

FOLLOWING COMPONENT
  SearchEngine facade wired to QueryProcessor and BM25Retriever
```

## 12. Logical Build Sequence

### MUST BUILD NOW

#### Step 1: Restore the lexical import boundary

- **File:** `retieval/lexical/retriever.py`
- **Why now:** It prevents `BM25Retriever` and the existing lexical example from importing.
- **Available dependencies:** `CandidateGenerator`, `InvertedIndex`, existing abstract retriever contract.
- **Unlocks:** Importable lexical retrieval modules.
- **Test:** Import and instantiate the lexical retriever boundary.
- **Blocking issue:** Reconcile the existing `retieval` spelling without creating a second package.

#### Step 2: Align BM25Retriever with BM25

- **File:** `retieval/lexical/bm25_retriever.py`
- **Why now:** Its scorer invocation does not match the existing BM25 methods.
- **Available dependencies:** `CandidateGenerator`, `BM25.score_document`.
- **Unlocks:** Ranked lexical results.
- **Test:** Index three documents and assert ranked integer IDs and numeric scores.
- **Blocking issue:** Preserve the existing lexical tuple contract.

#### Step 3: Add a focused lexical integration test

- **Location:** `tests/`
- **Why now:** Existing examples cannot verify the BM25 path.
- **Flow:** `text -> Analyzer -> IndexBuilder -> QueryProcessor -> CandidateGenerator -> BM25Retriever`.
- **Test:** Assert candidate set, ranked result shape, and deterministic ordering.
- **Blocking issue:** No project-local test configuration is present, but `unittest` is available.

#### Step 4: Wire the search facade

- **File:** `search/search_engine.py`
- **Why now:** It is the intended public lexical entry point but calls nonexistent methods.
- **Available dependencies:** `Query`, `QueryProcessor`, `CandidateGenerator`, working lexical retriever.
- **Unlocks:** Raw-text search through one facade.
- **Test:** Exercise supported mode behavior and ranked output.
- **Blocking issue:** Existing facade annotation returns `set[int]`, while BM25 retrieval returns ranked tuples; this contract must be resolved explicitly.

#### Step 5: Restore evaluation package importability

- **File:** `evaluations/__init__.py` plus existing misspelled metric filenames.
- **Why now:** Test discovery currently fails at package import.
- **Available dependencies:** Existing metric functions and `Evaluator` implementation.
- **Unlocks:** Retrieval-quality measurement.
- **Test:** Evaluate a known ranked list against known relevant IDs.
- **Blocking issue:** Evaluation uses string IDs while lexical retrieval uses integer IDs.

### BUILD AFTER CORE PIPELINE

1. Add formal tests for Analyzer and index invariants.
2. Add tests for BM25, TF-IDF, and specialized query operators.
3. Connect query parsing and expansion after basic search is stable.
4. Define an explicit ID conversion policy.
5. Connect Reranker and ScoreFusion after retriever output contracts stabilize.

### LATER / OPTIONAL

1. Connect HNSW to a dense retrieval execution path.
2. Add persistence round-trip tests.
3. Repair and test graph storage/traversal.
4. Implement a concrete hybrid fusion strategy.
5. Add RAG and planning layers.

### DO NOT BUILD YET

- Additional embedding providers.
- RAG generation.
- Graph-to-query matching.
- ANN optimization.
- Production persistence backends.
- A second package named `retrieval`.

## 13. Do Not Rebuild

| Existing component | File                              | Reuse reason                                                               |
| ------------------ | --------------------------------- | -------------------------------------------------------------------------- |
| Analyzer           | `analysis/analyzer.py`            | Existing pipeline is already used by indexing and query processing         |
| IndexBuilder       | `index/builder.py`                | Already populates forward, positional, inverted, and vocabulary structures |
| CandidateGenerator | `retieval/candidate_generator.py` | Existing OR candidate API is executable and manually checked               |
| BM25               | `ranking/bm25.py`                 | Existing scorer already reads forward index and statistics                 |
| TF-IDF             | `ranking/tfidf.py`                | Existing standalone scorer                                                 |
| VectorStore        | `vector/vector_store.py`          | Exact search and CRUD are covered by passing tests                         |
| VectorSimilarity   | `vector/similarity.py`            | Already shared by VectorStore and HNSW                                     |
| HNSW               | `vector/hnsw.py`                  | Existing ANN implementation has passing tests                              |
| DenseRetriever     | `retieval/dense/retriever.py`     | Existing raw-text delegation path                                          |
| Evaluation metrics | `evaluations/`                    | Metric implementations exist; package wiring is the issue                  |
| Reranker           | `ranking/reranker.py`             | Existing score-map sorting API                                             |
| ScoreFusion        | `ranking/fusion.py`               | Existing weighted integer-ID score-map API                                 |

## 14. Final Engineering State

**Current project state:** Several partially independent implementations exist. The vector/HNSW subsystem is the strongest verified area; the lexical subsystem is not integrated end to end.

**Core pipeline state:** Analysis, indexing, candidate generation, BM25, and TF-IDF exist separately. No verified public path produces ranked lexical results from raw query text.

**Biggest blocker:** Broken module wiring and incompatible contracts at the lexical retrieval boundary.

**Next file to build:** `tfidf_search_engine/retieval/lexical/retriever.py`

**Unknown areas:** Provider configuration, NLTK/spaCy resources, intended ID conversion, intended `RetrievalResult`, persistence runtime, graph intent, data-file usage, and the purpose of empty planning/RAG directories.

CURRENT_NEXT_FILE:
`tfidf_search_engine/retieval/lexical/retriever.py`

NEXT_ACTION:
Make the existing lexical retriever contract importable under the actual `retieval` package, then align `BM25Retriever` with the existing BM25 scoring API.

NEXT_TEST:
Build a three-document index and run `BM25Retriever.retrieve(["machine", "learning"])`, asserting ranked integer document IDs and numeric scores.

NEXT_5_STEPS:

1. Repair the lexical retriever import boundary.
2. Align `BM25Retriever` with the existing `BM25.score_document` contract.
3. Add a focused lexical end-to-end test.
4. Wire `SearchEngine` to the working lexical retriever.
5. Repair evaluation package imports and measure retrieval quality.
