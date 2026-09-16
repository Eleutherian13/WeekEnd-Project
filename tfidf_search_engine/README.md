# Retrieval Engine

A from-scratch, terminal-based retrieval engine that combines lexical, dense,

graph, hybrid, reranking, and RAG components into a single search pipeline.

The project is built around a modular retrieval architecture rather than a

single search algorithm. Documents are loaded from `data/documents.json`,

processed into searchable representations, retrieved through one or more

retrieval strategies, optionally reranked, and finally supplied as grounded

context to a local LLM.

## Current Status

The application provides an interactive terminal menu for the currently

configured retrieval capabilities:

```text

--------------------------------

Retrieval Engine

--------------------------------

Documents loaded: 15

Indexed documents: 15

Search engine ready.

1. Lexical BM25 - AVAILABLE

2. Dense Retrieval - AVAILABLE

3. Graph Retrieval - AVAILABLE

4. Hybrid Retrieval - AVAILABLE (lexical + dense)

5. Hybrid + Reranking - AVAILABLE

6. RAG / Answer - AVAILABLE

7. Exit

The system supports:

                    ┌─────────────────┐

                    │     Query       │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │ Query Analysis  │

                    │  & Planning     │

                    └────────┬────────┘

                             │

              ┌──────────────┼──────────────┐

              ▼              ▼              ▼

          ┌────────┐     ┌────────┐     ┌────────┐

          │ BM25   │     │ Dense  │     │ Graph  │

          │ Lexical│     │ Vector │     │ Search │

          └────┬───┘     └────┬───┘     └────┬───┘

               │              │              │

               └──────────────┼──────────────┘

                              ▼

                    ┌─────────────────┐

                    │ Hybrid Fusion   │

                    │      RRF        │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │ CrossEncoder    │

                    │   Reranking     │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │ Evidence        │

                    │ Construction    │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │ Context Builder │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │ Local Generator │

                    │     Ollama      │

                    └────────┬────────┘

                             │

                             ▼

                    ┌─────────────────┐

                    │   RAGResponse   │

                    └─────────────────┘

The CLI is an application layer over the repository's retrieval and search

components. The same underlying components can also be used programmatically

through SearchEngine.

End-to-End Verification

Verification performed on 2026-09-16.

Component   Status  Verification

Document loading    VERIFIED    data/documents.json loaded successfully

In-memory indexing  VERIFIED    All 15 documents indexed at startup

Lexical BM25 retrieval  VERIFIED    Real queries returned ranked document results

Dense retrieval VERIFIED    Local embedding model produced vector-based retrieval results

Graph retrieval VERIFIED    Graph entities and relations were used to retrieve connected documents

Hybrid retrieval    VERIFIED    Retrieval candidates were combined through Reciprocal Rank Fusion

CrossEncoder reranking  VERIFIED    Candidate ordering was rescored using a local CrossEncoder

Evidence construction   VERIFIED    Retrieved evidence remained traceable to source documents

Context construction    VERIFIED    Retrieved evidence was converted into RAG context

Ollama generation   VERIFIED    qwen2.5:7b generated answers from supplied RAG context

Missing provider/model handling VERIFIED    Unsupported or unavailable configuration fails explicitly

Full RAG application path   VERIFIED    Retrieval → reranking → evidence → context → generation is wired into the CLI

The complete test suite passed with 99 tests.

Real queries were used to exercise lexical, dense, graph, and hybrid retrieval,

followed by reranking, evidence construction, context construction, and local

Ollama generation.

The system is designed to fail explicitly when a required provider or model is

unavailable rather than silently fabricating retrieval or generation results.

Core Retrieval Pipeline

The main retrieval architecture is:

Query

  │

  ▼

QueryAnalyzer

  │

  ▼

QueryClassifier

  │

  ▼

QueryPlanner

  │

  ├──► Lexical Retrieval

  │        └──► BM25

  │

  ├──► Dense Retrieval

  │        └──► Vector Search

  │

  └──► Graph Retrieval

           └──► Graph Traversal

  │

  ▼

Candidate Fusion

  │

  ▼

Reciprocal Rank Fusion (RRF)

  │

  ▼

CrossEncoder Reranking

  │

  ▼

EvidenceBuilder

  │

  ▼

ContextBuilder

  │

  ▼

Generator

  │

  ▼

RAGResponse

Not every query necessarily needs every retrieval source. The planner can select

retrieval modes based on the characteristics of the query and the configured

retrieval components.

Retrieval Modes

1. Lexical BM25

The lexical retriever uses the repository's analyzer and inverted-index

infrastructure to retrieve documents using BM25 scoring.

Typical strengths:

Exact terms

Rare keywords

Identifiers

Technical terminology

Error codes and document-specific vocabulary

Example:

CVE-2025-1234

2. Dense Retrieval

Dense retrieval converts queries and documents into embeddings and retrieves

semantically similar documents using vector similarity.

This allows the system to retrieve relevant documents even when the query does

not use the exact wording found in the document.

Example:

How do computers learn patterns from examples?

3. Graph Retrieval

Graph retrieval uses document/entity relationships to traverse connected

information.

Graph metadata can represent entities and relations such as:

Machine Learning

      │

      ▼

Neural Networks

      │

      ▼

Computer Vision

      │

      ▼

Satellite Imagery

The graph layer is integrated into the retrieval pipeline and can return

documents associated with matched or traversed graph entities.

4. Hybrid Retrieval

Hybrid retrieval combines retrieval signals from multiple retrieval systems.

The current hybrid path combines:

BM25

  +

Dense Retrieval

  ↓

Reciprocal Rank Fusion

This allows lexical and semantic retrieval signals to contribute to the final

candidate set.

5. Hybrid + Reranking

After candidate generation and fusion, a local CrossEncoder can score the

query-document pairs again.

BM25 + Dense

      ↓

     RRF

      ↓

Candidate Set

      ↓

CrossEncoder

      ↓

Final Ranking

The CrossEncoder score is a model score and should not be interpreted as a

confidence percentage.

6. RAG / Answer

The RAG path combines retrieval and generation:

Query

  ↓

Retrieval

  ↓

Reranking

  ↓

Evidence

  ↓

Context

  ↓

Ollama

  ↓

Answer + Evidence

The generator receives retrieved context rather than independently searching

the corpus.

Project Structure

retrieval_engine/

│

├── analysis/

│   ├── tokenizer.py

│   ├── normalizer.py

│   ├── stopwords.py

│   ├── stemmer.py

│   ├── lemmatization.py

│   └── analyzer.py

│

├── document/

│   ├── document.py

│   ├── corpus.py

│   └── chunk.py

│

├── index/

│   ├── vocabulary.py

│   ├── posting.py

│   ├── posting_list.py

│   ├── inverted_index.py

│   ├── forward_index.py

│   ├── positional_index.py

│   ├── statistics.py

│   └── builder.py

│

├── query/

│   ├── query.py

│   ├── query_parser.py

│   ├── query_processing.py

│   ├── boolean_query.py

│   ├── phrase_query.py

│   ├── fuzzy_query.py

│   ├── wildcard_query.py

│   └── query_expansion.py

│

├── retrieval/

│   ├── candidate_generator.py

│   ├── lexical/

│   │   ├── retriever.py

│   │   └── bm25_retriever.py

│   ├── dense/

│   │   ├── retriever.py

│   │   └── vector_retriever.py

│   └── graph/

│       ├── retriever.py

│       └── graph_retriever.py

│

├── ranking/

│   ├── tfidf.py

│   ├── cosine.py

│   ├── bm25.py

│   ├── fusion.py

│   └── reranker.py

│

├── vector/

│   ├── embedding.py

│   ├── vector_store.py

│   ├── similarity.py

│   ├── hnsw.py

│   └── persistence.py

│

├── graph/

│   ├── node.py

│   ├── edge.py

│   ├── graph.py

│   ├── adjacency.py

│   ├── traversal.py

│   └── store.py

│

├── hybrid/

│   ├── fusion.py

│   ├── rrf.py

│   └── hybrid_retriever.py

│

├── planning/

│   ├── query_analyzer.py

│   ├── query_classifier.py

│   └── query_planner.py

│

├── reranking/

│   ├── cross_encoder.py

│   └── reranker.py

│

├── rag/

│   ├── retriever.py

│   ├── evidence.py

│   ├── context_builder.py

│   └── generator.py

│

├── evaluation/

│   ├── precision.py

│   ├── recall.py

│   ├── mrr.py

│   ├── map.py

│   ├── ndcg.py

│   └── evaluator.py

│

├── search/

│   └── search_engine.py

│

├── tests/

│

├── data/

│   └── documents.json

│

└── app.py

Architecture Responsibilities

analysis/

Responsible for text preprocessing:

Tokenization

Normalization

Stopword handling

Stemming

Lemmatization

document/

Contains the core document and corpus data models.

index/

Implements the lexical indexing infrastructure:

Vocabulary

Posting lists

Inverted index

Forward index

Positional index

Corpus statistics

query/

Represents and processes user queries.

retrieval/

Contains retrieval interfaces and retrieval implementations for lexical,

dense, and graph search.

ranking/

Contains standalone ranking algorithms and ranking-related components,

including BM25 and fusion utilities.

vector/

Contains the vector-search infrastructure:

Embedding interfaces

Vector store

Similarity calculations

HNSW index

Persistence utilities

graph/

Contains graph primitives and graph storage/traversal infrastructure.

hybrid/

Combines retrieval outputs from multiple retrieval sources.

planning/

Analyzes queries and creates retrieval plans.

reranking/

Performs second-stage ranking using CrossEncoder-based scoring.

rag/

Handles:

Retrieval

Evidence construction

Context construction

Generation

search/

Contains the SearchEngine orchestration facade.

The main programmatic interfaces are:

SearchEngine.search()

SearchEngine.retrieve()

SearchEngine.answer()

app.py

Provides:

Startup

Document loading

Index construction

Component initialization

Retrieval-mode selection

Interactive CLI input/output

Graceful shutdown

Document Data

The application reads:

data/documents.json

The top-level value must be a JSON array.

Each document contains:

{

  "document_id": 1,

  "text": "Machine learning is amazing",

  "metadata": {

    "topic": "machine learning"

  }

}

Required fields:

document_id: unique non-negative integer

text: string

Optional field:

metadata: JSON object

The metadata can also provide the entities and relationships used by the graph

retrieval integration.

Malformed records are rejected with validation errors rather than silently

skipped.

The current verification corpus contains 15 documents designed to exercise

different retrieval paths, including lexical lookup, semantic retrieval,

vector search, graph relationships, hybrid fusion, reranking, RAG, query

planning, indexing, and provenance.

The in-memory indexes are rebuilt whenever the application starts.

Requirements

Python 3.10+ is recommended.

Core dependencies include:

nltk

sentence-transformers

ollama

nltk is used by the default Porter stemmer.

sentence-transformers is used for dense embeddings and local CrossEncoder

reranking.

The Ollama Python integration is used only for local LLM generation.

The default lexical retrieval path does not require an external database or

remote API.

Dense retrieval and reranking require their local model dependencies and model

artifacts to be available.

Installation

From the repository root:

cd tfidf_search_engine

Create a virtual environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

The default NLTK processing path uses PorterStemmer, so an NLTK corpus

download is not required for the normal lexical CLI path.

Ollama Configuration

The RAG generator uses Ollama for local generation.

The generator can read:

OLLAMA_MODEL

OLLAMA_HOST

from the environment.

An explicitly configured local model can also be supplied directly through the

application's generator configuration.

The application reports provider and generation failures explicitly when

Ollama or the configured model is unavailable.

Example model:

qwen2.5:7b

Running the Application

From the project directory:

python app.py

The application initializes the corpus and reports the number of documents

loaded and indexed.

Example:

--------------------------------

Retrieval Engine

--------------------------------

Documents loaded: 15

Indexed documents: 15

Search engine ready.

You can then select a retrieval mode from the menu.

Example lexical query:

Enter query (or 'exit'/'quit'): CVE-2025-1234

The application returns ranked documents and scores.

Example:

1. Document: ...

   Score: ...

BM25 scores are retrieval scores. They are not probabilities, confidence

percentages, or explanations of relevance.

Use:

exit

or:

quit

to stop the application.

Blank queries are rejected and the application continues running.

Unknown terms return:

No results.

Example Queries

BM25

CVE-2025-1234

Useful for testing exact-term lexical retrieval.

Dense Retrieval

How do computers learn patterns from examples?

Useful for testing semantic similarity.

Graph Retrieval

How is machine learning connected to computer vision?

Useful for testing entity matching and graph traversal.

Hybrid Retrieval

How do keyword and semantic retrieval work together?

Useful for testing combined lexical and dense retrieval.

Hybrid + Reranking

Explain how retrieval candidates are improved before RAG.

Useful for testing candidate fusion followed by CrossEncoder reranking.

RAG

Explain how lexical, dense, and graph retrieval work together.

Useful for exercising the complete retrieval-to-generation pipeline.

Testing

Run the complete test suite:

python -m unittest discover -s tests -v

Run application-layer tests:

python -m unittest -v tests.test_app

Compile all Python files:

python -m compileall -q .

Check dependency consistency:

python -m pip check

The current repository verification includes tests covering the application's

document loading, indexing, SearchEngine orchestration, lexical retrieval,

dense retrieval, graph retrieval, hybrid fusion, reranking, RAG components,

indexing infrastructure, and vector-related functionality.

Design Principles

The project is intentionally built as separate retrieval and orchestration

layers instead of hiding the entire system behind a single high-level library.

The architecture separates:

Representation

      ↓

Indexing

      ↓

Retrieval

      ↓

Fusion

      ↓

Reranking

      ↓

Evidence

      ↓

Context

      ↓

Generation

This makes individual components testable and allows retrieval strategies to be

combined without replacing their underlying implementations.

Current Limitations

The current application has several deliberate limitations:

The primary user interface is terminal-based.

Indexes are rebuilt in memory when the application starts.

The CLI does not currently persist its runtime indexes between sessions.

Dense retrieval depends on locally available embedding models.

CrossEncoder reranking depends on a locally available reranking model.

Ollama generation depends on a locally available Ollama installation/model.

The graph currently depends on graph-compatible metadata being available in

the document corpus.

The project contains additional subsystem implementations that are not all

exposed as dedicated CLI modes yet.

These limitations concern application scope and configuration rather than the

existence of the underlying subsystem implementations.

Roadmap

Potential future work includes:

Persistent index storage

More advanced query expansion

Additional retrieval strategies

More comprehensive retrieval evaluation datasets

Retrieval-quality benchmarking

Additional application interfaces

Production deployment and service APIs

Broader graph construction strategies

More robust retrieval observability and tracing

These items are future work and are not represented as current capabilities.

License

This project is released under the MIT License.

See LICENSE.



### One important correction

I would **not** keep this old wording:

> “A small, terminal-based lexical search application...”

That undersells what you've actually built. Your current project is much closer to:

> **“A from-scratch retrieval engine combining lexical, dense, graph, hybrid fusion, reranking, and RAG components.”**

Also, the old README had a direct contradiction:

```text