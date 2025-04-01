#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import warnings
import copy
import importlib
from abc import ABC, abstractmethod


class BaseEmbeddings(ABC):
    """LangChain-like embedding function interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        raise NotImplementedError()

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Serialize Embeddings.

        :return: serializes this Embeddings so that it can be reconstructed by ``from_dict`` class method.
        :rtype: dict
        """
        return {"__class__": self.__class__.__name__, "__module__": self.__module__}

    @classmethod
    def from_dict(cls, data: dict) -> BaseEmbeddings | None:
        """Deserialize ``BaseEmbeddings`` into a concrete one using arguments.

        :return: concrete Embeddings or None if data is incorrect
        :rtype: BaseEmbeddings | None
        """
        data = copy.deepcopy(data)
        if isinstance(data, dict):
            class_type = data.pop("__class__", None)
            module_name = data.pop("__module__", None)

            if module_name:
                module = importlib.import_module(module_name)

                if class_type:
                    try:
                        cls = getattr(module, class_type)
                    except AttributeError:
                        raise AttributeError(
                            f"Module: {module} has no attribute {class_type}"
                        )

                    if cls:
                        with warnings.catch_warnings(record=True):
                            warnings.simplefilter("ignore", category=DeprecationWarning)
                            return cls(**data)

        return None
