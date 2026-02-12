"""Embedding service for generating vector representations of metadata artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.extractors.base import MetadataArtifact

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings from metadata artifacts.

    Uses sentence-transformers models to convert text into dense vector
    representations suitable for semantic search.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        cache_dir: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", "mps")
            cache_dir: Directory to cache downloaded models
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {device}")

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load the model
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(self.cache_dir) if self.cache_dir else None,
        )

        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for model {model_name}")
        self.dimension: int = dimension
        logger.info(f"Model loaded successfully (dimension: {self.dimension})")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            Matrix of embeddings (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )

    def embed_artifact(self, artifact: MetadataArtifact) -> np.ndarray:
        """Generate embedding for a metadata artifact.

        Args:
            artifact: Metadata artifact to embed

        Returns:
            Embedding vector
        """
        text = artifact.to_embedding_text()
        return self.embed_text(text)

    def embed_artifacts(self, artifacts: list[MetadataArtifact]) -> np.ndarray:
        """Generate embeddings for multiple artifacts in batch.

        Args:
            artifacts: List of metadata artifacts

        Returns:
            Matrix of embeddings (num_artifacts x embedding_dim)
        """
        if not artifacts:
            return np.array([])

        texts = [a.to_embedding_text() for a in artifacts]
        logger.info(f"Generating embeddings for {len(texts)} artifacts...")

        embeddings = self.embed_texts(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string.

        Alias for embed_text for consistency with common embedding service APIs.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.embed_text(query)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batch.

        Alias for embed_texts for consistency with common embedding service APIs.

        Args:
            texts: List of input texts

        Returns:
            Matrix of embeddings (num_texts x embedding_dim)
        """
        return self.embed_texts(texts)

    def get_dimension(self) -> int:
        """Get the dimensionality of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 768 for all-mpnet-base-v2)
        """
        return self.dimension
