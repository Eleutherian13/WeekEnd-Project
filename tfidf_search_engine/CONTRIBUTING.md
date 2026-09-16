# Contributing

Thank you for contributing to the Retrieval Engine.

## Scope

Keep changes focused on the existing repository structure and verified behavior. The supported application path is the terminal lexical BM25 workflow launched by `python app.py`.

Avoid presenting the repository's dense, graph, hybrid, reranking, planning, or RAG source trees as integrated application features unless the integration is implemented and tested end to end.

## Development Setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Before Opening a Pull Request

Run the complete project test suite:

```powershell
python -m unittest discover -s tests -v
```

Compile the Python source:

```powershell
python -m compileall -q .
```

Run the application and exercise at least one matching query and one no-result query:

```powershell
python app.py
```

Confirm that the documented commands still work from the `tfidf_search_engine` directory.

## Tests

Use behavior-level tests for application changes. The application-layer tests belong in `tests/test_app.py`; lower-level component tests should remain focused on their own contracts and should not be duplicated in application tests.

For CLI changes, test observable output and shutdown behavior where practical. Do not test only that a function was called.

## Data Changes

The application data format is a JSON array of records with:

- a unique non-negative integer `document_id`;
- string `text`;
- optional object `metadata`.

Do not add secrets, private documents, generated artifacts, or credentials to the repository.

## Code Style

- Preserve existing public APIs unless a change is necessary and documented.
- Prefer small, readable changes.
- Do not hide unexpected exceptions with broad exception handlers.
- Do not add unsupported claims, benchmarks, models, or integrations to documentation.
- Keep generated files and Python caches out of commits.

## Pull Requests

Describe:

- what changed;
- why the change was needed;
- tests and commands run;
- any remaining limitations or follow-up work.
