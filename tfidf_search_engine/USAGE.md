# Retrieval Engine Usage Guide

This guide explains how to install, run, populate, test, and troubleshoot the current retrieval engine application.

## What the application does

The application is a terminal-based lexical search engine. It loads documents from `data/documents.json`, builds an in-memory search index, and lets you search the documents interactively.

The runtime path is:

```text
data/documents.json
    -> Document objects
    -> Corpus
    -> Analyzer
    -> IndexBuilder
    -> BM25 scorer
    -> BM25Retriever
    -> SearchEngine
    -> ranked terminal results
```

The terminal application presents a capability-aware menu. Lexical BM25 is
always available. Dense retrieval, hybrid fusion, CrossEncoder reranking, and
RAG are enabled only when their existing local components initialize. Graph
retrieval is reported as not configured because no graph data source is
connected to `documents.json`.

The verified programmatic orchestration path is available through
`SearchEngine.retrieve()` and `SearchEngine.answer()`:

```text
raw query
  -> QueryPlanner
  -> lexical/dense/graph retrieval
  -> HybridRetriever / ReciprocalRankFusion
  -> CrossEncoder reranking
  -> EvidenceBuilder
  -> ContextBuilder
  -> Generator
  -> RAGResponse
```

`RAGResponse` exposes final retrieval results, source evidence, formatted
context, and generated answer separately. The generator receives only the
`RAGContext`; it does not retrieve documents independently.

## Requirements

- Python 3.10 or newer is recommended.
- Run commands from the `tfidf_search_engine` directory.
- The current application uses the Python standard library, repository-local modules, `nltk`, `sentence-transformers`, and `ollama` declared in `requirements.txt`.

Check the Python version:

```powershell
python --version
```

## Start the application

From the repository root:

```powershell
cd tfidf_search_engine
python app.py
```

The interactive menu is:

```text
1. Lexical BM25
2. Dense Retrieval
3. Graph Retrieval
4. Hybrid Retrieval
5. Hybrid + Reranking
6. RAG / Answer
7. Exit
```

Each mode is labeled `AVAILABLE`, `NOT CONFIGURED`, or `NOT IMPLEMENTED`.
Unavailable modes are not executed. Select a mode, enter a query, and provide
a positive `top_k` where requested.

The application prints a startup banner and reports the number of loaded and indexed documents:

```text
--------------------------------
Retrieval Engine
--------------------------------
Documents loaded: 6
Indexed documents: 6
Search engine ready.
```

It then prompts for a query:

```text
Enter query (or 'exit'/'quit'):
```

## Search interactively

Enter ordinary text queries. The query is analyzed using the same analyzer used while indexing the documents. Matching documents are returned in BM25 ranking order.

Example:

```text
Enter query (or 'exit'/'quit'): machine

1. Document: Machine learning is amazing
   Score: 1.1601

2. Document: Machine learning uses neural networks
   Score: 0.9255
```

A multi-term query searches for documents containing any of the analyzed terms in the default OR mode:

```text
Enter query (or 'exit'/'quit'): machine learning

1. Document: Machine learning is amazing
   Score: 1.9411

2. Document: Machine learning uses neural networks
   Score: 1.5486

3. Document: Deep learning powers artificial intelligence
   Score: 0.6231
```

The displayed score is the BM25 score returned by the existing retrieval implementation. It is not a percentage or confidence value.

## Exit the application

Use either command:

```text
exit
```

or:

```text
quit
```

The application exits with:

```text
Goodbye.
```

You can also use `Ctrl+C` or send EOF. These are handled as clean shutdowns.

## Input behavior

### Blank input

Whitespace-only input is rejected without stopping the application:

```text
Enter query (or 'exit'/'quit'):
Please enter a query.
```

The prompt is shown again.

### No matching documents

A query with no matching terms returns:

```text
No results.
```

The application continues accepting queries.

### Repeated queries

Queries can be entered repeatedly during the same session. Each query is processed independently against the already-built in-memory index.

## Document data format

Documents are stored in `data/documents.json` as a JSON array. Each record must be an object containing:

- `document_id`: a non-negative integer
- `text`: the document text as a string
- `metadata`: an optional JSON object

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

The `document_id` values must be unique. The current terminal output displays the document `text` and returned BM25 `score`; metadata is loaded into the document model but is not displayed by the CLI.

After changing `data/documents.json`, restart `python app.py` so the corpus and indexes are rebuilt.

## How indexing works

At startup, `app.py` performs the following composition steps:

1. `load_corpus()` reads and validates the JSON records.
2. Each valid record becomes a `Document`.
3. Each `Document` is added to a `Corpus`.
4. `Analyzer` tokenizes, normalizes, removes stopwords, and stems or lemmatizes text.
5. `IndexBuilder` adds every document to the forward, positional, inverted, and vocabulary indexes.
6. `BM25` is configured with the builder's forward index and corpus statistics.
7. `BM25Retriever` combines candidate generation with BM25 scoring.
8. `SearchEngine` processes raw user queries and delegates retrieval.
9. The CLI resolves returned document IDs through the `Corpus` and displays document text and scores.

The indexes are in memory. They are rebuilt each time the application starts.

## Validation and data errors

The loader reports malformed data instead of silently skipping it. Examples of invalid data include:

- The JSON file is not valid JSON.
- The top-level JSON value is not a list.
- A record is not an object.
- `document_id` or `text` is missing.
- `document_id` is not a valid integer.
- `text` is not a string.
- `metadata` is not an object.
- A document ID is duplicated.

Correct the reported record or JSON file and restart the application.

An empty JSON array is valid:

```json
[]
```

The application starts with an empty corpus and reports that no documents are available for searching. Queries then return no results.

## Run tests

Run the complete test suite from the `tfidf_search_engine` directory:

```powershell
python -m unittest discover -s tests -v
```

The application-layer tests are in `tests/test_app.py` and
`tests/test_cli_modes.py`. They verify corpus loading, engine construction,
menu dispatch, real lexical output, capability reporting, invalid input,
unavailable graph handling, and clean exit.

The lower-level tests cover analysis, indexing, retrieval, ranking, graph, hybrid, and vector components.

## Compile the repository

Compile all Python files to check syntax:

```powershell
python -m compileall -q .
```

No output and exit code `0` indicate successful compilation.

## Verify dependencies

The project declares the `nltk` dependency used by the default Porter stemmer. You can check the active Python environment with:

```powershell
python -m pip check
```

A successful check prints:

```text
No broken requirements found.
```

## Programmatic End-to-End Retrieval and RAG

The complete verified pipeline is available from the menu when its components
are configured. Programmatic callers can also construct `SearchEngine` with
the independently verified dense and graph retrievers, a `Reranker`, a
document text provider, and a `Generator`. Call `retrieve(query, top_k)` to
inspect planner-selected candidates without generation, or call
`answer(query, top_k)` to receive a `RAGResponse`.

The response fields provide separate inspection points:

- `results`: final reranked `RetrievalResult` objects.
- `evidence`: source document IDs and text resolved by `EvidenceBuilder`.
- `context`: the bounded `RAGContext` passed to the generator.
- `answer`: generated text, or `None` when retrieval has no results.

For local Ollama generation, configure an installed model explicitly:

```python
from rag import OllamaGenerator

generator = OllamaGenerator(model_name="qwen2.5:7b", timeout=120)
```

Alternatively set `OLLAMA_MODEL` and optionally `OLLAMA_HOST`. Missing models,
provider errors, timeouts, invalid responses, and empty contexts raise explicit
errors. The generator never invents a response when Ollama is unavailable.

The graph retriever currently requires graph nodes and edges to be supplied by
the caller. It is not automatically populated from `data/documents.json`.

## Verification Status

Verified on 2026-09-16:

| Area                             | Status                                                    |
| -------------------------------- | --------------------------------------------------------- |
| Documents and indexing           | VERIFIED                                                  |
| Lexical retrieval                | VERIFIED                                                  |
| Dense retrieval                  | VERIFIED with local Sentence Transformers                 |
| Graph retrieval                  | VERIFIED with a programmatically supplied graph           |
| Hybrid RRF fusion                | VERIFIED                                                  |
| CrossEncoder reranking           | VERIFIED with local Sentence Transformers                 |
| Evidence and context provenance  | VERIFIED                                                  |
| Ollama generation                | VERIFIED with local `qwen2.5:7b`                          |
| Ollama environment configuration | NOT CONFIGURED; explicit model configuration was used     |
| Full RAG path in `app.py`        | VERIFIED; unavailable components remain explicitly marked |

The complete suite passed 99 tests. Real queries confirmed deterministic
retrieval, justified reranking, source-traceable evidence, grounded answers,
no-result behavior, and explicit failure for an unavailable Ollama model.

## Troubleshooting

### `python` cannot find `app.py`

Change into the application directory first:

```powershell
cd path\to\funProjectWeekend\tfidf_search_engine
python app.py
```

### The application reports invalid JSON

Open `data/documents.json` and confirm that it contains a valid JSON array. Use double quotes around JSON keys and string values.

### A document does not appear in results

Check that:

- The document is present in `data/documents.json`.
- Its `document_id` is unique.
- Its `text` contains terms related to the query.
- The application was restarted after editing the data file.

The analyzer normalizes and stems terms, so a query does not need to exactly match the original capitalization or punctuation.

### Search returns `No results.`

This means no indexed document matched the analyzed query terms. Try a term that appears in the document text, such as `machine`, `learning`, `cats`, or `neural`.

## Current limitations

- The index is rebuilt on every startup.
- The application is terminal-only.
- The CLI displays document text and retrieval scores; RAG also displays source evidence.
- Modes that require unavailable models or providers are explicitly marked not configured.
- There is no persistence layer for the built index in the application workflow.
- The full planner/dense/graph/RAG orchestration is programmatic; `app.py` does not yet construct it.
- Graph data is not derived automatically from `data/documents.json`.
