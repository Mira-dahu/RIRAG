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
        # 检索相关段落
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
        answer = generator.generate_answer(question, [p["PassageText"] for p in retrieved_passages_info[:20]], documents)
        results.append({
            "QuestionID": question_id,
            "Question": question,
            "Rankings": rankings,
            "PassageInfo": retrieved_passages_info,
            "Answer": answer
        })
    return results
def main():

    data_loader = DataLoader("data/raw")
    retriever = HybridRetriever()
    formatter = OutputFormatter()

    #openai.api_key = ""
    #openai.api_base = ""
    deepseek_api_key = ""
    generator = AnswerGenerator(deepseek_api_key)
    #generator = AnswerGenerator(openai.api_key)

    documents = data_loader.load_documents()
    unseen_questions = data_loader.load_unseen_questions()

    passages = []
    passage_ids = []
    for doc_id, doc_passages in documents.items():
        for p in doc_passages:
            passages.append(p["text"])
            passage_ids.append(p["ID"])

    print("Building index...")
    retriever.index_passages(passages)

    print("Processing unseen questions...")
    results = process_unseen_questions(
        retriever,
        generator,
        unseen_questions,
        passages,
        passage_ids,
        documents
    )

    formatter.save_trec_format(results, "submissions/subtask1_results5.8.txt")
    formatter.save_json_format(results, "submissions/subtask2_results5.8.json")
    print("Results saved to submissions/ directory")

if __name__ == "__main__":
    main()