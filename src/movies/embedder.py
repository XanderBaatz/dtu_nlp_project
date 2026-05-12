from abc import abstractmethod
from typing import Protocol

import numpy as np
from openai import OpenAI
from pydantic_settings import BaseSettings  # noqa: TC002
from sentence_transformers import SentenceTransformer

from movies.config import Settings

settings = Settings()


class Embedder(Protocol):
    """Vector embedder abstract class."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...


class OpenAIEmbedder(Embedder):
    """OpenAI LLM vector embedder."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key or settings.CAMPUSAI_API_KEY,
            base_url=api_url or settings.CAMPUSAI_API_URL,
        )
        self.model = model or settings.CAMPUSAI_EMBED_MODEL

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        batch_size = settings.BATCH_SIZE
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend(e.embedding for e in response.data)

        arr = np.array(all_embeddings, dtype=np.float32)
        return self._normalize(arr)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=[query],
        )

        arr = np.array(
            [response.data[0].embedding],
            dtype=np.float32,
        )

        return self._normalize(arr)[0]

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms


class LocalEmbedder(Embedder):
    """Local LLM vector embedder."""

    def __init__(
        self,
        model_name: str = settings.LOCAL_EMBED_MODEL,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = settings.BATCH_SIZE

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


def get_embedder(cfg: BaseSettings | None = None) -> Embedder:
    """Instantiate the configured embedder (local or OpenAI-compatible API)."""
    cfg = cfg or settings
    if cfg.LOCAL_EMBEDDING:
        return LocalEmbedder(cfg.LOCAL_EMBED_MODEL)
    return OpenAIEmbedder(
        api_key=cfg.CAMPUSAI_API_KEY,
        api_url=cfg.CAMPUSAI_API_URL,
        model=cfg.CAMPUSAI_EMBED_MODEL,
    )
