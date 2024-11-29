import torch
import openai
from typing import List, Dict
from src.data_processing.data_loader import DataLoader
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.output_formatter import OutputFormatter
import openai
from typing import List, Dict

def process_unseen_questions(
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        questions: List[Dict],
        passages: List[str],
        passage_ids: List[str],
        documents: Dict[str, List[Dict]]
) -> List[Dict]:
    results = []
    for question_item in questions:
        question_id = question_item["QuestionID"]
        question = question_item["Question"]
        retrieved = retriever.retrieve(question, top_k=10)
        rankings = []
        retrieved_passages_info = []
        for passage, score in retrieved:
            passage_idx = passages.index(passage)
            passage_id = passage_ids[passage_idx]
            passage_doc_info = [doc_passage for doc_id, doc_passages in documents.items() for doc_passage in doc_passages if doc_passage["ID"] == passage_id][0]
            retrieved_passages_info.append({
                "ID": passage_id,
                "DocumentID": passage_doc_info["DocumentID"],
                "PassageID": passage_doc_info["PassageID"],
                "PassageText": passage_doc_info["text"]
            })
            rankings.append((passage_id, score))
        # 生成答案
        answer = generator.generate_answer(question, [p["PassageText"] for p in retrieved_passages_info[:10]], documents,
                                            question_id)
        results.append({
            "QuestionID": question_id,
            "Question": question,
            "Rankings": rankings,
            "PassageInfo": retrieved_passages_info,
            "Answer": answer
        })
    return results
