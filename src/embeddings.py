from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper around a sentence-transformers model for document and query embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)
