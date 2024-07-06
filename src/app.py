from src.data_loader import load_contracts
from src.pipeline import create_pipeline

def main():
    contracts = load_contracts()
    qa_pipeline = create_pipeline(contracts)

    question = "What are the key terms of this contract?"
    result = qa_pipeline({"query": question})

    print(result["result"])

if __name__ == "__main__":
    main()