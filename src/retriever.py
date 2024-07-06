from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from src.config import OPENAI_API_KEY

def create_retriever(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_creator = VectorstoreIndexCreator(embedding=embeddings)
    index = index_creator.from_documents(documents)
    return index.vectorstore.as_retriever()