import torch
from src.data_processing.data_loader import DataLoader
from src.data_processing.llm_data_processor import LLMDataProcessor  # Import the data processor
from src.models.llm_trainer import LLMTrainer  # Import the trainer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.output_formatter import OutputFormatter
import openai
from typing import List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer  # Add this import

from transformers import AutoModelForCausalLM, AutoTokenizer  # Ensure this import is included


def train_llm():
    try:
        # Initialize data processor and trainer
        data_processor = LLMDataProcessor()
        llm_trainer = LLMTrainer()

        # Load evaluation data
        excel_path = os.path.join("data", "raw", "1-446(1).xlsx")
        logger.info(f"Loading evaluation data from {excel_path}")
        eval_data = data_processor.load_evaluation_data(excel_path)

        if eval_data is None or eval_data.empty:
            logger.error("Failed to load evaluation data")
            return

        # Prepare training samples
        logger.info("Preparing training samples")
        training_samples = data_processor.prepare_training_samples(eval_data)

        if not training_samples:
            logger.error("No training samples generated")
            return

        # Prepare training data
        logger.info("Preparing training data")
        training_data = llm_trainer.prepare_training_data(training_samples)

        # Train model
        logger.info("Starting model training")

        # Load the model with trust_remote_code=True

        llm_trainer.model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm3-6b",
            trust_remote_code=True  # Trust remote code for loading the model
        )
        llm_trainer.train(training_data)

        # Save model
        model_save_path = "D:/trained_llm"
        os.makedirs(model_save_path, exist_ok=True)
        llm_trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

    except Exception as e:
        logger.error(f"Error in train_llm: {str(e)}")


# The rest of your main function remains unchanged...

def process_unseen_questions(
        retriever: HybridRetriever,
        generator: AnswerGenerator,
        questions: List[Dict],
        passages: List[str],
        passage_ids: List[str],
        documents: Dict[str, List[Dict]]
) -> List[Dict]:
    """Process unseen questions and generate results."""
    results = []
    for question_item in questions:
        question_id = question_item["QuestionID"]
        question = question_item["Question"]

        # Retrieve relevant passages
        retrieved = retriever.retrieve(question, top_k=10)

        # Get passage IDs and scores
        rankings = []
        retrieved_passages_info = []

        for passage, score in retrieved:
            passage_idx = passages.index(passage)
            passage_id = passage_ids[passage_idx]

            # Get full passage information
            passage_doc_info = [doc_passage for doc_id, doc_passages in documents.items()
                                for doc_passage in doc_passages if doc_passage["ID"] == passage_id][0]
            retrieved_passages_info.append({
                "ID": passage_id,
                "DocumentID": passage_doc_info["DocumentID"],
                "PassageID": passage_doc_info["PassageID"],
                "PassageText": passage_doc_info["text"]
            })
            rankings.append((passage_id, score))

        # Generate answer
        answer = generator.generate_answer_with_llm(question, [p["PassageText"] for p in retrieved_passages_info[:20]],
                                           documents)

        results.append({
            "QuestionID": question_id,
            "Question": question,
            "Rankings": rankings,
            "PassageInfo": retrieved_passages_info,
            "Answer": answer
        })

    return results

def main():
    try:
        logger.info("Starting main process")

        # 1. Train LLM
        logger.info("Starting LLM training")
        train_llm()

        # 2. Initialize components
        data_loader = DataLoader("data/raw")
        retriever = HybridRetriever()

        # Set OpenAI API key and base URL
        openai.api_key = "sk-0q7L6AoETlqIbXYBB9Bb237313Ee445a8141AcD56a2aD0D9"
        openai.api_base = "https://api.mixrai.com/v1"

        # 初始化AnswerGenerator，同时传入API密钥和不传入（使用训练好的LLM）
        # 初始化AnswerGenerator，同时传入API密钥和不传入（使用训练好的LLM）
        generator_openai = AnswerGenerator(openai.api_key)
        generator_llm = AnswerGenerator("D:/trained_llm")

        # 3. Load data
        logger.info("Loading documents and unseen questions")
        documents = data_loader.load_documents()
        unseen_questions = data_loader.load_unseen_questions()

        # Prepare retrieval data
        passages = []
        passage_ids = []

        for doc_id, doc_passages in documents.items():
            for p in doc_passages:
                passages.append(p["text"])
                passage_ids.append(p["ID"])

        # Build index
        logger.info("Building index...")
        retriever.index_passages(passages)

        # 4. Process unseen questions and generate answers using OpenAI
      #  logger.info("Processing unseen questions with OpenAI...")
       # results_openai = process_unseen_questions(
       #     retriever,
       #     generator_openai,
        #    unseen_questions,
         #   passages,
          #  passage_ids,
       #     documents
        #)

        # 5. Save results from OpenAI

     #   formatter.save_trec_format(results_openai, "submissions/subtask1_results_openai.txt")
#        formatter.save_json_format(results_openai, "submissions/subtask2_results_openai.json")

 #       logger.info("Results from OpenAI saved to submissions/ directory")

        # 6. Process unseen questions and generate answers using LLM
        logger.info("Processing unseen questions with LLM...")
        results_llm = process_unseen_questions(
            retriever,
            generator_llm,
            unseen_questions,
            passages,
            passage_ids,
            documents
        )

        # 7. Save results from LLM
        formatter = OutputFormatter()
        formatter.save_trec_format(results_llm, "submissions/subtask1_results_llm.txt")
        formatter.save_json_format(results_llm, "submissions/subtask2_results_llm.json")

        logger.info("Results from LLM saved to submissions/ directory")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()