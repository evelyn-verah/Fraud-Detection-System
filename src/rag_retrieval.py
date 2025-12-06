from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings


def build_vector_store(texts, embeddings: Embeddings) -> FAISS:
    """Build a FAISS vector store from a list of texts."""
    docs = [Document(page_content=t) for t in texts]
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store
