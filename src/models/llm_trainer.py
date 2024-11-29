import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json


class LLMTrainer:
    def __init__(self, model_name: str = "THUDM/chatglm3-6b"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_training_data(self, samples: List[Dict]) -> List[Dict]:
        """准备训练数据"""
        training_data = []

        for sample in samples:
            # 构建输入输出对
            input_text = f"问题: {sample['question']}\n\n相关文档: {' '.join(sample['context'])}"
            output_text = sample['answer']

            # Tokenize
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            output_ids = self.tokenizer(output_text, return_tensors="pt").input_ids

            training_data.append({
                "input_ids": input_ids,
                "labels": output_ids,
                "attention_mask": torch.ones_like(input_ids)
            })

        return training_data

    def train(self, training_data: List[Dict],
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              batch_size: int = 4):
        """训练模型"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]

                # 准备批次数据
                input_ids = torch.cat([item["input_ids"] for item in batch])
                labels = torch.cat([item["labels"] for item in batch])
                attention_mask = torch.cat([item["attention_mask"] for item in batch])

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(training_data)}")

    def save_model(self, path: str):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)