from src.data_processing.data_loader import DataLoader
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.output_formatter import OutputFormatter
import openai
from typing import List, Dict
from src.generation.fine_tuner import DeepSeekFineTuner, LegalQADataset
import pandas as pd


def fine_tune_model(data_loader, documents):
    # 加载adjust.xlsx数据
    adjust_df = pd.read_excel("data/raw/adjust2.xlsx")

    # 初始化微调器
    fine_tuner = DeepSeekFineTuner(
        model_path="deepseek-ai/deepseek-llm-7b-base",
        training_args={
            "output_dir": "./fine_tuned_model",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
        }
    )

    # 准备数据集
    questions, contexts, answers, quality_scores = fine_tuner.prepare_dataset(
        adjust_df,
        documents
    )

    # 创建训练集
    train_dataset = LegalQADataset(
        questions[:int(len(questions) * 0.8)],
        contexts[:int(len(contexts) * 0.8)],
        answers[:int(len(answers) * 0.8)],
        fine_tuner.tokenizer
    )

    # 创建验证集
    eval_dataset = LegalQADataset(
        questions[int(len(questions) * 0.8):],
        contexts[int(len(contexts) * 0.8):],
        answers[int(len(answers) * 0.8):],
        fine_tuner.tokenizer
    )

    # 开始训练
    fine_tuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quality_scores=quality_scores
    )

    return "./fine_tuned_model"
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
        # 获取段落ID（这里的ID是之前定义的文档中的ID）和分数
        rankings = []
        retrieved_passages_info = []
        for passage, score in retrieved:
            passage_idx = passages.index(passage)
            passage_id = passage_ids[passage_idx]
            # 获取完整的段落信息
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
def main():
    data_loader = DataLoader("data/raw")
    retriever = HybridRetriever()
    formatter = OutputFormatter()
    documents = data_loader.load_documents()
    unseen_questions = data_loader.load_unseen_questions()

    fine_tuned_model_path = fine_tune_model(data_loader, documents)

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
    formatter.save_trec_format(results, "submissions/subtask1_results6.1.txt")
    formatter.save_json_format(results, "submissions/subtask2_results6.1.json")
    print("Results saved to submissions/ directory")

if __name__ == "__main__":
    main()