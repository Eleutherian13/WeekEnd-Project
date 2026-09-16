from __future__ import annotations

from abc import ABC, abstractmethod

from .context_builder import RAGContext


class Generator(ABC):
    """
    Abstract interface for answer generation.

    Concrete implementations may use:

    - a local LLM
    - an API-based LLM
    - another text-generation backend
    """

    @abstractmethod
    def generate(
        self,
        context: RAGContext,
    ) -> str:
        """
        Generate an answer from grounded RAG context.

        Implementations should use the supplied context as the
        evidence available for answering the query.
        """

        if not isinstance(context, RAGContext):
            raise TypeError(
                "context must be a RAGContext"
            )

        raise NotImplementedError

    def __call__(
        self,
        context: RAGContext,
    ) -> str:
        """Allow the generator to be called like a function."""
        return self.generate(context)   