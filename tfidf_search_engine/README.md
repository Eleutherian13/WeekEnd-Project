# Retrieval Engine

A small, terminal-based lexical search application built from the repository's document, analysis, indexing, BM25 ranking, and retrieval components.

The verified application loads documents from `data/documents.json`, builds an in-memory index, and accepts interactive queries through `app.py`.

## Current Status

The terminal application exposes the verified retrieval modes through an
interactive menu. Lexical BM25 is always available; dense, hybrid, reranking,
and RAG entries are shown only when their configured local components load.
Graph retrieval remains unavailable unless graph data is supplied by an
existing graph-building integration.

```text
data/documents.json
    -> Document
    -> Corpus
    -> Analyzer
    -> IndexBuilder
    -> BM25
    -> BM25Retriever
    -> SearchEngine
    -> ranked CLI output
```

The programmatic and CLI paths reuse the repository's vector, graph, hybrid,
reranking, planning, and RAG components. The CLI does not create graph nodes
from `documents.json`.

The programmatic `SearchEngine` also exposes a verified orchestrated path when
the relevant components are configured:

```text
raw query
  -> QueryPlanner
  -> lexical/dense/graph retrieval
  -> ReciprocalRankFusion
  -> CrossEncoder reranking
  -> EvidenceBuilder
  -> ContextBuilder
  -> Generator
  -> RAGResponse
```

Use `SearchEngine.retrieve()` for planner-selected candidates and
`SearchEngine.answer()` for a response containing retrieved results, evidence,
context, and an optional generated answer. The CLI is a thin adapter over
these APIs.

## End-to-End Verification

Verification performed on 2026-09-16:

| Component                               | Status         | Evidence                                                                                    |
| --------------------------------------- | -------------- | ------------------------------------------------------------------------------------------- |
| Document loading and in-memory indexing | VERIFIED       | `data/documents.json` loaded and indexed 6 documents.                                       |
| Lexical BM25 retrieval                  | VERIFIED       | Real CLI queries returned deterministic ranked results and `No results.` for unknown terms. |
| Dense retrieval                         | VERIFIED       | Local Sentence Transformers embeddings returned valid deterministic results.                |
| Graph retrieval                         | VERIFIED       | Query-matched graph entities traversed to document nodes with distance scores.              |
| Hybrid fusion                           | VERIFIED       | Real candidates were unified through the existing RRF implementation.                       |
| CrossEncoder reranking                  | VERIFIED       | Local CrossEncoder scores changed ordering when justified by scores.                        |
| Evidence and context                    | VERIFIED       | Evidence IDs and text traced back to corpus documents.                                      |
| Ollama generation                       | VERIFIED       | `qwen2.5:7b` generated answers from supplied RAG context.                                   |
| Missing model/provider handling         | VERIFIED       | Missing configuration and unknown models failed explicitly.                                 |
| Environment-based Ollama configuration  | NOT CONFIGURED | `OLLAMA_MODEL` and `OLLAMA_HOST` were unset; explicit model configuration was used.         |
| Full RAG wiring in `app.py`             | VERIFIED       | The menu invokes retrieval, reranking, evidence, context, and generation when configured.   |

The complete test suite passed with 99 tests. Real programmatic queries
exercised lexical, dense, graph, and hybrid retrieval, followed by reranking,
evidence construction, context construction, and Ollama answer generation.
Unavailable providers did not produce fabricated answers.

Remaining blockers are configuration and data-source limitations rather than
broken components: graph nodes are supplied programmatically rather than built
from `documents.json`, and environment-based Ollama configuration is optional
even though explicit local model selection is supported.

## Capabilities

The current CLI can:

- Load and validate JSON documents.
- Build an in-memory lexical index at startup.
- Normalize, filter, and stem query/document terms through the existing analyzer.
- Retrieve and rank matching documents with BM25.
- Display document text and returned BM25 scores.
- Handle repeated queries, blank input, no-result queries, `exit`, `quit`, EOF, and Ctrl+C.
- Support the existing `SearchEngine` `top_k` and retrieval-mode APIs when used programmatically.

## Requirements

- Python 3.10 or newer is recommended.
- `nltk` is required by the default Porter stemmer used by the application.
- `sentence-transformers` is required for the verified dense and reranking integrations.
- `ollama` is required only when using the local Ollama RAG generator.
- No external database, API key, or downloaded embedding model is required for the verified CLI path.

## Installation

From the repository root, enter this project directory:

```powershell
cd tfidf_search_engine
```

Create and activate a virtual environment if desired:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application dependency:

```powershell
python -m pip install -r requirements.txt
```

The application currently uses NLTK's built-in `PorterStemmer`, so no NLTK corpus download is required for the default CLI path.

The Ollama generator reads `OLLAMA_MODEL` and optionally `OLLAMA_HOST` from the
environment. It raises an explicit provider or generation error when Ollama or
the configured model is unavailable; it does not fabricate an answer.

## Run the Application

From `tfidf_search_engine`:

```powershell
python app.py
```

The menu reports each mode as `AVAILABLE`, `NOT CONFIGURED`, or
`NOT IMPLEMENTED`. Select a mode, enter a query, and provide a positive
`top_k`. RAG output includes the answer and source document evidence.

Startup reports the loaded and indexed document counts:

```text
--------------------------------
Retrieval Engine
--------------------------------
Documents loaded: 6
Indexed documents: 6
Search engine ready.
```

Enter a query at the prompt:

```text
Enter query (or 'exit'/'quit'): machine learning
```

Example output:

```text
1. Document: Machine learning is amazing
   Score: 1.9411

2. Document: Machine learning uses neural networks
   Score: 1.5486
```

The score shown is the BM25 score returned by the existing retriever. It is not a confidence percentage or relevance explanation.

Use `exit` or `quit` to stop. Blank input is rejected with a prompt to enter a query. Unknown terms display `No results.` and the application continues.

## Document Data

The application reads `data/documents.json`. The top-level value must be a JSON array. Each record must contain:

- `document_id`: a unique non-negative integer.
- `text`: a string used for indexing and display.
- `metadata`: an optional JSON object stored on the `Document`.

Example:

```json
[
  {
    "document_id": 1,
    "text": "Machine learning is amazing",
    "metadata": {
      "topic": "machine learning"
    }
  },
  {
    "document_id": 2,
    "text": "Cats chase mice",
    "metadata": {
      "topic": "animals"
    }
  }
]
```

Malformed records are rejected with a validation error; invalid documents are not silently skipped. An empty array is valid and starts an empty searchable corpus.

The index is rebuilt in memory each time `app.py` starts. Restart the application after changing the data file.

## Architecture

- `document/`: `Document` and `Corpus` data models.
- `analysis/`: tokenization, normalization, stopword removal, and stemming/lemmatization.
- `index/`: forward, inverted, positional, vocabulary, and statistics indexes.
- `ranking/`: BM25 and other standalone ranking components.
- `retieval/`: the repository's existing retrieval package name and lexical retrievers.
- `query/`: raw query and query-processing objects.
- `search/`: the `SearchEngine` orchestration facade.
- `app.py`: application startup, corpus loading, component construction, CLI input, output, and shutdown.
- `tests/`: unit and application-layer tests.
- `search/search_engine.py`: preserves the legacy lexical `search()` API and adds the planner-driven `retrieve()` and `answer()` orchestration APIs.

The package directory is currently named `retieval`; this spelling is preserved by the active imports and is not renamed by the CLI application.

## Testing

Run all tests in the project test directory:

```powershell
python -m unittest discover -s tests -v
```

Run the application-layer tests only:

```powershell
python -m unittest -v tests.test_app
```

Compile all Python files:

```powershell
python -m compileall -q .
```

Check installed package consistency:

```powershell
python -m pip check
```

The documented test command covers the repository's `tests/` suite, including application, SearchEngine orchestration, lexical, dense, graph, hybrid, reranking, RAG, indexing, and vector-related tests present in that directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the focused contribution and validation workflow.

Before submitting a change:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
python app.py
```

## Limitations and Roadmap

Current limitations:

- The verified application is terminal-only.
- Indexes are rebuilt on every startup and are not persisted by the CLI workflow.
- The CLI displays document text and BM25 scores; it does not display metadata.
- The active application path exposes lexical, dense, hybrid, reranking, and RAG modes when their local dependencies are configured.
- Graph retrieval is not available in the CLI because no graph-building source is connected to `documents.json`.
- The repository contains additional experimental or standalone subsystem code that is not integrated into this CLI workflow.
- The retrieval package name is currently `retieval`, including its active imports.

Possible future work, subject to design and validation, includes persistent indexes, a supported package-name migration, broader application interfaces, and deliberate integration of additional retrieval modes. These are not current capabilities.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
