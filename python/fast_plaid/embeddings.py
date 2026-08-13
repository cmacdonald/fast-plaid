from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch


class Embeddings(ABC):
    """Abstract base class for document embedding collections.

    Subclass this to provide lazy or disk-backed access to embeddings without
    materialising the entire collection into memory.  The ``fast_plaid_rust``
    backend calls :meth:`__len__` and :meth:`__getitem__` (and optionally
    :meth:`get_batch`) to retrieve tensors on demand, so only the tensors
    needed for the current processing batch are ever loaded.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of documents."""

    @abstractmethod
    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the embedding tensor for document *index*.

        Args:
        ----
        index:
            Zero-based document index.

        Returns:
        -------
        A 2-D tensor of shape ``(num_tokens, embedding_dim)``.

        """

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a stacked tensor for the given document indices.

        The default implementation calls :meth:`__getitem__` for each index
        and concatenates the results along the first dimension.  Subclasses
        may override this for more efficient batch loading.

        Args:
        ----
        indices:
            Ordered list of zero-based document indices.

        Returns:
        -------
        A 2-D tensor of shape ``(total_tokens, embedding_dim)``.

        """
        tensors = [self[i] for i in indices]
        return torch.cat(tensors, dim=0)


class ListEmbeddings(Embeddings):
    """Thin :class:`Embeddings` wrapper around a plain :class:`list` or sequence.

    This is used internally to provide backward-compatibility: when the caller
    passes a ``list[torch.Tensor]`` or any other ``Sequence[torch.Tensor]``,
    it is wrapped in a :class:`ListEmbeddings` so that the Rust backend
    receives an object that satisfies the :class:`Embeddings` protocol.

    Args:
    ----
    tensors:
        A sequence of 2-D tensors, one per document.

    """

    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        self._tensors = tensors

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._tensors[index]

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        return torch.cat([self._tensors[i] for i in indices], dim=0)


__all__ = ["Embeddings", "ListEmbeddings"]
