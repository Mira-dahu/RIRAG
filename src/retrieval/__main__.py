import json
from pathlib import Path
from src.retrieval.evaluator import Evaluator

def main():
    evaluator = Evaluator()

    base_path = "data/raw/StructuredRegulatoryDocuments"
    test_data_path = f"{base_path}/ObliQA_test.json"

    print("Evaluating retrieval performance...")
    recall_10 = evaluator.evaluate_retrieval(test_data_path)

    print(f"Results:")
    print(f"Recall@10: {recall_10:.4f}")

if __name__ == "__main__":
    main()