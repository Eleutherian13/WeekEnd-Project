# Retrieval Engine

A small, terminal-based lexical search application built from the repository's document, analysis, indexing, BM25 ranking, and retrieval components.

The verified application loads documents from `data/documents.json`, builds an in-memory index, and accepts interactive queries through `app.py`.

## Current Status

The supported end-to-end workflow is lexical BM25 retrieval:

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

The repository also contains vector, graph, hybrid, reranking, evaluation, planning, and RAG source trees. They are not part of the verified terminal application workflow and should not be treated as production-ready integrations.

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
- No external database, model server, API key, or downloaded embedding model is required for the verified CLI path.

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

## Run the Application

From `tfidf_search_engine`:

```powershell
python app.py
```

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

The documented test command covers the repository's `tests/` suite, including application, SearchEngine, lexical retrieval, indexing, graph, hybrid, and vector-related tests present in that directory.

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
- The active application path uses lexical BM25 retrieval only.
- The repository contains additional experimental or standalone subsystem code that is not integrated into this CLI workflow.
- The retrieval package name is currently `retieval`, including its active imports.

Possible future work, subject to design and validation, includes persistent indexes, a supported package-name migration, broader application interfaces, and deliberate integration of additional retrieval modes. These are not current capabilities.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
