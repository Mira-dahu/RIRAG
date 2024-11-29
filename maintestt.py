from src.data_processing.data_loader import DataLoader
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.output_formatter import OutputFormatter
from typing import List, Dict
import numpy as np
from sklearn.metrics import ndcg_score




def calculate_recall_at_k(retrieved_passages: List[str], relevant_passages: List[str], k: int = 10) -> float:
    if not relevant_passages:
        return 0.0
    retrieved_set = set(get_passage_prefix(p) for p in retrieved_passages[:k])
    relevant_set = set(get_passage_prefix(p) for p in relevant_passages)
    return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)


def calculate_mrr(retrieved_passages: List[str], relevant_passages: List[str]) -> float:
    if not relevant_passages:
        return 0.0
    relevant_prefixes = set(get_passage_prefix(p) for p in relevant_passages)
    for i, passage in enumerate(retrieved_passages, 1):
        if get_passage_prefix(passage) in relevant_prefixes:
            return 1.0 / i
    return 0.0


def calculate_ndcg_at_k(retrieved_passages: List[str], relevant_passages: List[str], k: int = 10) -> float:
    if not relevant_passages:
        return 0.0

    relevant_prefixes = set(get_passage_prefix(p) for p in relevant_passages)
    relevance_scores = []
    for passage in retrieved_passages[:k]:
        relevance_scores.append(1.0 if get_passage_prefix(passage) in relevant_prefixes else 0.0)

    while len(relevance_scores) < k:
        relevance_scores.append(0.0)

    ideal_relevance = [1.0] * len(relevant_passages) + [0.0] * (k - len(relevant_passages))
    ideal_relevance = ideal_relevance[:k]

    return ndcg_score([ideal_relevance], [relevance_scores])


def process_unseen_questions(
        retriever: HybridRetriever,
        questions: List[Dict],
        passages: List[str],
        passage_ids: List[str],
        documents: Dict[str, List[Dict]]
) -> List[Dict]:
    results = []
    for question_item in questions:
        question_id = question_item["QuestionID"]
        question = question_item["Question"]
        # 检索相关段落
        retrieved = retriever.retrieve(question, top_k=10)
        # 获取段落ID和分数
        rankings = []
        retrieved_passages_info = []
        for passage, score in retrieved:
            passage_idx = passages.index(passage)
            passage_id = passage_ids[passage_idx]
            passage_doc_info = [doc_passage for doc_id, doc_passages in documents.items()
                                for doc_passage in doc_passages if doc_passage["ID"] == passage_id][0]
            retrieved_passages_info.append({
                "ID": passage_id,
                "DocumentID": passage_doc_info["DocumentID"],
                "PassageID": passage_doc_info["PassageID"],
                "PassageText": passage_doc_info["text"]
            })
            rankings.append((passage_id, score))

        results.append({
            "QuestionID": question_id,
            "Question": question,
            "Rankings": rankings,
            "PassageInfo": retrieved_passages_info
        })
    return results


def main():
    data_loader = DataLoader("data/raw")
    retriever = HybridRetriever()
    formatter = OutputFormatter()

    documents = data_loader.load_documents()
    test_questions = data_loader.load_obliqa_dataset("test")  # 加载测试集

    passages = []
    passage_ids = []
    for doc_id, doc_passages in documents.items():
        for p in doc_passages:
            passages.append(p["text"])
            passage_ids.append(p["ID"])

    print("Building index...")
    retriever.index_passages(passages)



    print("\nEvaluating retrieval performance on test set...")
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for question in test_questions:
        retrieved = retriever.retrieve(question["Question"], top_k=10)
        retrieved_passages = [p for p, _ in retrieved]

        relevant_passages = [p["Passage"] for p in question["Passages"]]

        recall_scores.append(calculate_recall_at_k(retrieved_passages, relevant_passages))
        mrr_scores.append(calculate_mrr(retrieved_passages, relevant_passages))
        ndcg_scores.append(calculate_ndcg_at_k(retrieved_passages, relevant_passages))

    print(f"Average Recall@10: {np.mean(recall_scores):.4f}")
    print(f"Average MRR: {np.mean(mrr_scores):.4f}")
    print(f"Average NDCG@10: {np.mean(ndcg_scores):.4f}")

    #formatter.save_trec_format(results, "submissions/subtask1_results5.8.txt")
    print("Results saved to submissions/ directory")


if __name__ == "__main__":
    main()