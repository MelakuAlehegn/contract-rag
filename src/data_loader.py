# data_loader.py
import os
from langchain_community.document_loaders import PyPDFLoader

def load_contracts(data_dir="data/contracts"):
    contracts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {filename}")
            contracts.extend(documents)
    return contracts