import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import os
import json

class LegalQADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]

        prompt = f"""Question: {question}\n\nContext: {context}\n\nAnswer:"""
        completion = f"{answer}</s>"

        # 构建输入
        encoded = self.tokenizer(
            prompt + completion,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = encoded["input_ids"].clone()
        # 将prompt部分的label设为-100
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[0, :prompt_len] = -100

        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels[0]
        }


class DeepSeekFineTuner:
    def __init__(self, model_path: str, training_args: Dict = None):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 默认训练参数
        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=5e-5,
            **training_args if training_args else {}
        )

    def prepare_dataset(self, adjust_df: pd.DataFrame, documents: Dict):
        questions = []
        contexts = []
        answers = []
        quality_scores = []

        for _, row in adjust_df.iterrows():
            question_id = row['QuestionID']
            lang = row['lang']

            # 从documents中获取相关上下文
            relevant_docs = self._get_relevant_documents(question_id, documents)
            if relevant_docs and lang == 1:  # 仅处理英文数据
                questions.append(row['Question'])
                contexts.append('\n'.join(relevant_docs))
                answers.append(row['Answer'])
                quality_scores.append(row['quality_score'])  # 假设adjust.xlsx中添加了quality_score列

        return questions, contexts, answers, quality_scores

    def _get_relevant_documents(self, question_id: str, documents: Dict) -> List[str]:
        """
        获取与问题相关的文档段落

        Args:
            question_id: 问题ID
            documents: 包含所有文档的字典，格式为 {doc_id: List[passage_dict]}

        Returns:
            List[str]: 相关文档段落的列表
        """
        relevant_texts = []

        # 从训练/开发/测试集中查找问题
        question_info = None
        for dataset_file in ['ObliQA_train.json', 'ObliQA_dev.json', 'ObliQA_test.json']:
            try:
                with open(os.path.join(self.data_dir, dataset_file), 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                    for item in dataset:
                        if item['QuestionID'] == question_id:
                            question_info = item
                            break
                    if question_info:
                        break
            except Exception as e:
                print(f"Error loading {dataset_file}: {e}")
                continue

        if not question_info:
            return relevant_texts

        # 获取问题相关的所有段落
        passages = question_info.get('Passages', [])

        # 按照PassageID的顺序收集相关文本
        for passage_info in passages:
            doc_id = str(passage_info['DocumentID'])
            passage_id = passage_info['PassageID']

            if doc_id in documents:
                # 在documents中查找对应的段落
                for doc_passage in documents[doc_id]:
                    if doc_passage['PassageID'] == passage_id:
                        relevant_texts.append(doc_passage['Passage'])
                        break

        return relevant_texts

    def train(self, train_dataset, eval_dataset=None, quality_scores=None):
        if quality_scores is not None:
            # 根据质量分数调整训练轮数
            avg_score = np.mean(quality_scores)
            if avg_score > 0.8:  # 如果平均质量分数大于0.8
                self.training_args.num_train_epochs += 2  # 增加训练轮数

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        # 保存微调后的模型
        self.model.save_pretrained("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")

    def evaluate(self, eval_dataset):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=eval_dataset,
        )

        metrics = trainer.evaluate()
        return metrics