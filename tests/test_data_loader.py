import os
from src.data_loader import load_contracts

def test_load_contracts():
    # Create a sample directory and file for testing
    test_dir = "tests/sample_contracts"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "sample_contract.pdf"), 'w') as f:
        f.write("Sample contract content.")

    contracts = load_contracts(test_dir)
    assert len(contracts) > 0
    assert isinstance(contracts[0], str)