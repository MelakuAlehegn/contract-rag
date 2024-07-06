from langchain.chains import RetrievalQA
from src.retriever import create_retriever
from src.generator import create_generator

def create_pipeline(documents):
    retriever = create_retriever(documents)
    generator = create_generator()
    qa_chain = RetrievalQA.from_chain_type(
        llm=generator, retriever=retriever, return_source_documents=True
    )
    return qa_chain