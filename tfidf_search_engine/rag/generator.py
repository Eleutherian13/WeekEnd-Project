from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .context_builder import RAGContext

try:
    import ollama
except ImportError:  # pragma: no cover - exercised by environments without Ollama
    ollama = None

try:
    import httpx
except ImportError:  # pragma: no cover - provided by the Ollama dependency
    httpx = None


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


class OllamaGenerator(Generator):
    """Generate grounded answers with a local Ollama model."""

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        configured_model = model_name or os.getenv(
            "OLLAMA_MODEL",
        )
        if not configured_model or not configured_model.strip():
            raise ValueError(
                "an Ollama model is required; set model_name or OLLAMA_MODEL"
            )

        if host is not None and not isinstance(host, str):
            raise TypeError("host must be a string or None")

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(
                timeout,
                (int, float),
            ):
                raise TypeError("timeout must be a number or None")

            if timeout <= 0:
                raise ValueError("timeout must be greater than zero")

        self.model_name = configured_model.strip()
        self.host = host or os.getenv("OLLAMA_HOST")
        self.timeout = timeout

        if client is not None:
            if not callable(getattr(client, "generate", None)):
                raise TypeError("client must provide a generate method")
            self._client = client
            return

        if ollama is None:
            raise RuntimeError(
                "Ollama provider is unavailable; install the ollama package"
            )

        try:
            client_options: dict[str, Any] = {}
            if self.timeout is not None:
                client_options["timeout"] = self.timeout

            self._client = ollama.Client(
                host=self.host,
                **client_options,
            )
        except Exception as error:
            raise RuntimeError(
                "Ollama provider could not be initialized"
            ) from error

    def generate(self, context: RAGContext) -> str:
        if not isinstance(context, RAGContext):
            raise TypeError("context must be a RAGContext")

        if not context.query.strip():
            raise ValueError("context query must not be empty")

        if not context.text.strip() or not context.evidence:
            raise ValueError("cannot generate an answer from empty context")

        prompt = self._build_prompt(context)

        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
            )
        except Exception as error:
            if self._is_timeout_error(error):
                raise TimeoutError(
                    "Ollama generation timed out"
                ) from error

            raise RuntimeError(
                f"Ollama model {self.model_name!r} is unavailable "
                "or failed during generation"
            ) from error

        answer = self._response_text(response)
        if not answer.strip():
            raise RuntimeError("Ollama returned an empty answer")

        return answer.strip()

    @staticmethod
    def _is_timeout_error(error: Exception) -> bool:
        if isinstance(error, TimeoutError):
            return True

        if httpx is not None and isinstance(
            error,
            httpx.TimeoutException,
        ):
            return True

        return "timeout" in error.__class__.__name__.casefold()

    @staticmethod
    def _build_prompt(context: RAGContext) -> str:
        return (
            "Answer the query using only the supplied evidence. "
            "Do not use outside knowledge. If the evidence does not "
            "support an answer, say that the evidence is insufficient.\n\n"
            f"Query: {context.query}\n\n"
            f"Evidence:\n{context.text}\n\n"
            "Answer:"
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, dict):
            answer = response.get("response")
        else:
            answer = getattr(response, "response", None)

        if not isinstance(answer, str):
            raise RuntimeError(
                "Ollama returned an invalid generation response"
            )

        return answer

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"host={self.host!r}, "
            f"timeout={self.timeout!r}"
            f")"
        )