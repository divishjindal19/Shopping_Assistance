#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from ibm_watsonx_ai.foundation_models.embeddings.base_embeddings import BaseEmbeddings
from ibm_watsonx_ai.wml_client_error import MissingExtension


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Embedding that utilizes sentence transformer, compatibile with ``RAGPattern``.

    Requires sentence_transformers to be installed by pip.

    :param model_name: name of the model from Huggingface
    :type model_name: str

    :param model_params: parameters given to the ``SentenceTransformer`` model constructor, defaults to None
    :type model_params: dict, optional

    :param encode_params: parameters given to the ``SentenceTransformer`` model ``encode`` method, defaults to None
    :type encode_params: dict, optional
    """

    def __init__(
        self,
        model_name: str,
        model_params: dict | None = None,
        encode_params: dict | None = None,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise MissingExtension("sentence_transformers")
        self.model_name = model_name
        self.model_params = model_params or {}
        self.encode_params = encode_params or {}
        self.model = SentenceTransformer(self.model_name, **self.model_params)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, **self.encode_params).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, **self.encode_params).tolist()

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "model_name": self.model_name,
                "model_params": self.model_params,
                "encode_params": self.encode_params,
            }
        )
        return d
