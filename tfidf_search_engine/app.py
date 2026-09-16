import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from graph.store import GraphStore
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from query.query_processing import QueryProcessor
from rag import OllamaGenerator
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.dense.retriever import DenseRetriever
from retieval.graph.graph_retriever import SimpleGraphRetriever
from retieval.lexical.bm25_retriever import BM25Retriever
from reranking import Reranker, SentenceTransformerCrossEncoder
from search.search_engine import (
    GenerationFailure,
    RetrievalFailure,
    SearchEngine,
)
from vector.embeddings.sentence_transformer import SentenceTransformerEmbedding


DATA_PATH = Path(__file__).with_name("data") / "documents.json"


@dataclass(frozen=True)
class Runtime:
    corpus: Corpus
    engine: SearchEngine
    dense_error: str | None = None
    graph_error: str | None = None
    reranker_error: str | None = None
    generator_error: str | None = None

    @property
    def dense_available(self) -> bool:
        return self.engine.dense_retriever is not None

    @property
    def graph_available(self) -> bool:
        return self.engine.graph_retriever is not None

    @property
    def reranker_available(self) -> bool:
        return self.engine.reranker is not None

    @property
    def generator_available(self) -> bool:
        return self.engine.generator is not None


def _normalize_graph_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    return str(value).strip()


def build_graph_from_corpus(corpus: Corpus) -> Graph:
    """Construct a graph from the document metadata held in the corpus."""

    graph = Graph()
    graph_store = GraphStore(graph)
    entity_metadata: dict[str, dict[str, Any]] = {}
    seen_edges: set[tuple[str, str, str, int]] = set()

    for document in corpus:
        document_id = document.document_id
        document_node_id = str(document_id)
        document_node = Node(
            document_node_id,
            "document",
            {
                "document_id": document_id,
                "text": document.text,
                "topic": document.metadata.get("topic"),
                "source": "documents.json",
            },
        )
        if not graph.has_node(document_node_id):
            graph.add_node(document_node)

        metadata = document.metadata or {}
        entity_names: list[str] = []
        raw_entities = metadata.get("entities", [])
        if isinstance(raw_entities, list):
            for raw_entity in raw_entities:
                entity_name = _normalize_graph_value(raw_entity)
                if entity_name:
                    entity_names.append(entity_name)

        for entity_name in entity_names:
            record = entity_metadata.setdefault(
                entity_name,
                {
                    "name": entity_name,
                    "document_ids": [],
                    "related_entities": [],
                    "relations": [],
                    "source_documents": [],
                },
            )
            if document_id not in record["document_ids"]:
                record["document_ids"].append(document_id)
            if document_id not in record["source_documents"]:
                record["source_documents"].append(document_id)

        for relation in metadata.get("relations", []):
            if not isinstance(relation, dict):
                continue

            relation_name = _normalize_graph_value(relation.get("relation"))
            source_name = _normalize_graph_value(relation.get("source"))
            target_name = _normalize_graph_value(relation.get("target"))

            if not relation_name or not target_name:
                continue

            if not source_name:
                source_name = entity_names[0] if entity_names else ""

            if not source_name or not target_name:
                continue

            source_record = entity_metadata.setdefault(
                source_name,
                {
                    "name": source_name,
                    "document_ids": [],
                    "related_entities": [],
                    "relations": [],
                    "source_documents": [],
                },
            )
            target_record = entity_metadata.setdefault(
                target_name,
                {
                    "name": target_name,
                    "document_ids": [],
                    "related_entities": [],
                    "relations": [],
                    "source_documents": [],
                },
            )

            if document_id not in source_record["document_ids"]:
                source_record["document_ids"].append(document_id)
            if document_id not in target_record["document_ids"]:
                target_record["document_ids"].append(document_id)
            if document_id not in source_record["source_documents"]:
                source_record["source_documents"].append(document_id)
            if document_id not in target_record["source_documents"]:
                target_record["source_documents"].append(document_id)

            source_record["relations"].append(
                {
                    "document_id": document_id,
                    "relation": relation_name,
                    "source": source_name,
                    "target": target_name,
                }
            )
            target_record["related_entities"].append(
                {
                    "document_id": document_id,
                    "relation": relation_name,
                    "source": source_name,
                    "target": target_name,
                }
            )

            seen_edges.add((source_name, target_name, relation_name, document_id))

        for entity_name in entity_names:
            if not graph.has_node(entity_name):
                graph.add_node(
                    Node(
                        entity_name,
                        "entity",
                        {
                            "name": entity_name,
                            "document_ids": [document_id],
                            "source_documents": [document_id],
                        },
                    )
                )

            if not graph.has_edge(entity_name, document_node_id):
                graph.add_edge(
                    Edge(
                        entity_name,
                        document_node_id,
                        "mentioned_in",
                        metadata={
                            "document_id": document_id,
                            "source": "metadata.entities",
                        },
                    )
                )

    for entity_name, record in entity_metadata.items():
        if graph.has_node(entity_name):
            continue
        graph.add_node(Node(entity_name, "entity", record))

    for source_name, target_name, relation_name, document_id in sorted(
        seen_edges,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    ):
        if not graph.has_node(source_name):
            graph.add_node(
                Node(source_name, "entity", {"name": source_name})
            )
        if not graph.has_node(target_name):
            graph.add_node(
                Node(target_name, "entity", {"name": target_name})
            )
        if not graph.has_edge(source_name, target_name):
            graph.add_edge(
                Edge(
                    source_name,
                    target_name,
                    relation_name,
                    metadata={
                        "document_id": document_id,
                        "source": source_name,
                        "target": target_name,
                    },
                )
            )

    return graph_store.graph


def load_corpus(path: Path = DATA_PATH) -> Corpus:
    try:
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of document records")

    corpus = Corpus()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"document record {position} must be an object"
            )

        missing_fields = {"document_id", "text"} - record.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"document record {position} is missing: {fields}"
            )

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"document record {position} metadata must be an object"
            )

        try:
            document = Document(
                document_id=record["document_id"],
                text=record["text"],
                metadata=metadata,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid document record {position}: {error}"
            ) from error

        try:
            corpus.add(document)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid document record {position}: {error}"
            ) from error

    return corpus


def create_search_engine(
    corpus: Corpus,
    configure_optional_components: bool = False,
) -> SearchEngine:
    analyzer = Analyzer()
    builder = IndexBuilder(
        analyzer=analyzer,
        vocabulary=Vocabulary(),
        inverted_index=InvertedIndex(),
    )

    for document in corpus:
        builder.add_document(document)

    retriever = BM25Retriever(
        candidate_generator=CandidateGenerator(builder.inverted_index),
        scorer=BM25(builder.forward_index, builder.statistics),
    )

    graph_retriever = None
    try:
        graph = build_graph_from_corpus(corpus)
        graph_retriever = SimpleGraphRetriever(
            graph,
            max_hops=2,
            document_node_type="document",
        )
    except Exception:
        graph_retriever = None

    if not configure_optional_components:
        return SearchEngine(
            query_processor=QueryProcessor(analyzer),
            lexical_retriever=retriever,
            graph_retriever=graph_retriever,
        )

    dense_retriever = None
    try:
        dense_retriever = DenseRetriever.from_documents(
            list(corpus),
            SentenceTransformerEmbedding(),
        )
    except Exception:
        dense_retriever = None

    reranker = None
    if dense_retriever is not None:
        try:
            reranker = Reranker(
                SentenceTransformerCrossEncoder(),
                lambda document_id: corpus[int(document_id)].text,
            )
        except Exception:
            reranker = None

    generator = None
    model_name = _configured_ollama_model()
    if model_name is not None:
        try:
            generator = OllamaGenerator(model_name=model_name)
        except Exception:
            generator = None

    return SearchEngine(
        query_processor=QueryProcessor(analyzer),
        lexical_retriever=retriever,
        dense_retriever=dense_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
        document_text_provider=lambda document_id: corpus[
            int(document_id)
        ].text,
        generator=generator,
    )


def create_runtime(corpus: Corpus) -> Runtime:
    """Build the CLI runtime and retain optional-component diagnostics."""

    dense_error = None
    graph_error = None
    reranker_error = None
    generator_error = None

    try:
        engine = create_search_engine(
            corpus,
            configure_optional_components=True,
        )
    except Exception as error:
        raise RuntimeError("unable to initialize the retrieval engine") from error

    if engine.dense_retriever is None:
        dense_error = (
            "Sentence Transformers embedding model is unavailable"
        )

    if engine.graph_retriever is None:
        graph_error = "graph metadata is not available or malformed"

    if engine.reranker is None:
        reranker_error = (
            "Sentence Transformers CrossEncoder model is unavailable"
        )

    if engine.generator is None:
        generator_error = (
            "Ollama is unavailable or no installed model is configured"
        )

    return Runtime(
        corpus=corpus,
        engine=engine,
        dense_error=dense_error,
        graph_error=graph_error,
        reranker_error=reranker_error,
        generator_error=generator_error,
    )


def _configured_ollama_model() -> str | None:
    model_name = os.getenv("OLLAMA_MODEL")
    if model_name and model_name.strip():
        return model_name.strip()

    try:
        import ollama

        models = ollama.list().models
    except Exception:
        return None

    if not models:
        return None

    return models[0].model


def _status(label: str, available: bool, reason: str | None = None) -> str:
    if available:
        return f"AVAILABLE{f' ({reason})' if reason else ''}"
    return f"NOT CONFIGURED{f' ({reason})' if reason else ''}"


def print_menu(runtime: Runtime) -> None:
    print("Status labels: AVAILABLE / NOT CONFIGURED / NOT IMPLEMENTED")
    print("\n1. Lexical BM25 - AVAILABLE")
    print(
        "2. Dense Retrieval - "
        + _status("dense", runtime.dense_available, runtime.dense_error)
    )
    print(
        "3. Graph Retrieval - "
        + _status("graph", runtime.graph_available, runtime.graph_error)
    )
    print(
        "4. Hybrid Retrieval - "
        + _status(
            "hybrid",
            runtime.dense_available or runtime.graph_available,
            "lexical + dense"
            if runtime.dense_available
            else "lexical + graph"
            if runtime.graph_available
            else "dense or graph retriever required",
        )
    )
    print(
        "5. Hybrid + Reranking - "
        + _status(
            "reranking",
            runtime.reranker_available
            and (runtime.dense_available or runtime.graph_available),
            runtime.reranker_error,
        )
    )
    print(
        "6. RAG / Answer - "
        + _status(
            "rag",
            runtime.reranker_available and runtime.generator_available,
            runtime.generator_error or runtime.reranker_error,
        )
    )
    print("7. Exit")


def _read_top_k() -> int | None:
    raw_value = input("Top_k (positive integer): ").strip()
    try:
        top_k = int(raw_value)
    except ValueError:
        print("Invalid top_k. Enter a positive integer.")
        return None

    if top_k <= 0:
        print("Invalid top_k. Enter a positive integer.")
        return None

    return top_k


def _read_query() -> str | None:
    query = input("Query: ").strip()
    if not query:
        print("Please enter a query.")
        return None
    return query


def _print_results(results: list[Any], corpus: Corpus) -> None:
    if not results:
        print("No results.")
        return

    for position, result in enumerate(results, start=1):
        document = corpus[int(result.document_id)]
        print(f"\n{position}. Document ID: {result.document_id}")
        print(f"   Text: {document.text}")
        print(f"   Score: {result.score:.4f}")


def _print_legacy_results(results: list[tuple[int, float]], corpus: Corpus) -> None:
    if not results:
        print("No results.")
        return

    for position, (document_id, score) in enumerate(results, start=1):
        print(f"\n{position}. Document ID: {document_id}")
        print(f"   Text: {corpus[document_id].text}")
        print(f"   Score: {score:.4f}")


def _run_mode(runtime: Runtime, selection: str) -> None:
    if selection == "3" and not runtime.graph_available:
        print(f"Graph Retrieval: NOT CONFIGURED. {runtime.graph_error}")
        return

    if selection == "2" and not runtime.dense_available:
        print(f"Dense Retrieval: NOT CONFIGURED. {runtime.dense_error}")
        return

    if selection == "4" and not (
        runtime.dense_available or runtime.graph_available
    ):
        print("Hybrid Retrieval: NOT CONFIGURED. Dense or graph retriever is unavailable.")
        return

    if selection == "5" and not (
        runtime.reranker_available
        and (runtime.dense_available or runtime.graph_available)
    ):
        print("Hybrid + Reranking: NOT CONFIGURED.")
        print(runtime.reranker_error or runtime.dense_error or runtime.graph_error)
        return

    if selection == "6" and not (
        runtime.reranker_available and runtime.generator_available
    ):
        print("RAG / Answer: NOT CONFIGURED.")
        print(runtime.generator_error or runtime.reranker_error)
        return

    query = _read_query()
    if query is None:
        return

    top_k = _read_top_k()
    if top_k is None:
        return

    try:
        if selection == "1":
            _print_legacy_results(
                runtime.engine.search(query, top_k=top_k),
                runtime.corpus,
            )
            return

        if selection == "2":
            results = runtime.engine.dense_retriever.retrieve(
                query,
                k=top_k,
            )
            _print_results(results, runtime.corpus)
            return

        if selection == "3":
            results = runtime.engine.graph_retriever.retrieve(
                query,
                top_k=top_k,
            )
            _print_results(results, runtime.corpus)
            return

        if selection == "4":
            plan = runtime.engine.planner.plan(query, top_k=top_k)
            results = runtime.engine.retrieve(query, top_k=top_k)
            print(
                "Modes used: "
                + ", ".join(mode.value for mode in plan.enabled_modes)
            )
            _print_results(results, runtime.corpus)
            return

        if selection == "5":
            plan = runtime.engine.planner.plan(query, top_k=top_k)
            candidates = runtime.engine.retrieve(query, top_k=top_k)
            reranked = runtime.engine.reranker.rerank(
                query,
                candidates,
                top_k=top_k,
            )
            original_scores = {
                result.document_id: result.score
                for result in candidates
            }
            print(
                "Modes used: "
                + ", ".join(mode.value for mode in plan.enabled_modes)
            )
            for result in reranked:
                document = runtime.corpus[int(result.document_id)]
                print(f"\nDocument ID: {result.document_id}")
                print(f"   Text: {document.text}")
                print(
                    f"   Retrieval score: "
                    f"{original_scores[result.document_id]:.4f}"
                )
                print(f"   Reranking score: {result.score:.4f}")
            return

        response = runtime.engine.answer(query, top_k=top_k)
        if response.answer is None:
            print("No results. No answer was generated.")
            return

        print(f"\nAnswer:\n{response.answer}")
        print("\nSources:")
        for evidence in response.evidence:
            print(f"- Document ID: {evidence.document_id}")
            print(f"  Evidence: {evidence.text}")
    except RetrievalFailure as error:
        print(f"Retrieval failed: {error}")
    except GenerationFailure as error:
        print(f"Generation failed: {error}")
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"Mode failed: {error}")


def run() -> None:
    print("--------------------------------")
    print("Retrieval Engine")
    print("--------------------------------")

    corpus = load_corpus()
    runtime = create_runtime(corpus)

    print(f"Documents loaded: {len(corpus)}")
    if not corpus:
        print("No documents available for searching.")
    else:
        print(f"Indexed documents: {len(corpus)}")
    print("Search engine ready.")

    try:
        while True:
            print_menu(runtime)
            selection = input("\nSelect mode: ").strip().lower()
            if selection in {"7", "exit", "quit"}:
                break
            if selection not in {"1", "2", "3", "4", "5", "6"}:
                print("Invalid mode. Select a number from 1 to 7.")
                continue
            _run_mode(runtime, selection)
    except EOFError:
        pass
    except KeyboardInterrupt:
        print()

    print("\nGoodbye.")


if __name__ == "__main__":
    run()
