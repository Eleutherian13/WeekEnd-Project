import json
from pathlib import Path

from analysis.analyzer import Analyzer
from document.corpus import Corpus
from document.documents import Document
from index.builder import IndexBuilder
from index.inverted_index import InvertedIndex
from index.vocabulary import Vocabulary
from query.query_processing import QueryProcessor
from ranking.bm25 import BM25
from retieval.candidate_generator import CandidateGenerator
from retieval.lexical.bm25_retriever import BM25Retriever
from search.search_engine import SearchEngine


DATA_PATH = Path(__file__).with_name("data") / "documents.json"


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


def create_search_engine(corpus: Corpus) -> SearchEngine:
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

    return SearchEngine(
        query_processor=QueryProcessor(analyzer),
        lexical_retriever=retriever,
    )


def run() -> None:
    print("--------------------------------")
    print("Retrieval Engine")
    print("--------------------------------")

    corpus = load_corpus()
    engine = create_search_engine(corpus)

    print(f"Documents loaded: {len(corpus)}")
    if not corpus:
        print("No documents available for searching.")
    else:
        print(f"Indexed documents: {len(corpus)}")
    print("Search engine ready.")

    try:
        while True:
            query = input("\nEnter query (or 'exit'/'quit'): ").strip()
            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                print("Please enter a query.")
                continue

            results = engine.search(query)
            if not results:
                print("No results.")
                continue

            for position, (document_id, score) in enumerate(results, start=1):
                document = corpus[document_id]
                print(f"\n{position}. Document: {document.text}")
                print(f"   Score: {score:.4f}")
    except EOFError:
        pass
    except KeyboardInterrupt:
        print()

    print("\nGoodbye.")


if __name__ == "__main__":
    run()
