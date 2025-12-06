from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.vectorstores import FAISS


def build_rag_chain(llm: LLM, vector_store: FAISS) -> RetrievalQA:
    """Create a RetrievalQA chain on top of a vector store."""
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )
    return qa
