import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np


class RetrievalDataset(Dataset):
    def __init__(self, queries, passages, labels, tokenizer, max_length=512):
        self.queries = queries
        self.passages = passages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        passage = self.passages[idx]
        label = self.labels[idx]

        query_encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        passage_encoding = self.tokenizer(
            passage,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'passage_input_ids': passage_encoding['input_ids'].squeeze(),
            'passage_attention_mask': passage_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class RetrievalTrainer:
    def __init__(self, model, train_dataset, val_dataset, device):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device

        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 2e-5
        self.warmup_steps = 1000

        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                passage_input_ids = batch['passage_input_ids'].to(self.device)
                passage_attention_mask = batch['passage_attention_mask'].to(self.device)

                query_embeddings = self.model(query_input_ids, query_attention_mask)
                passage_embeddings = self.model(passage_input_ids, passage_attention_mask)

                similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.t())

                labels = torch.arange(similarity_matrix.size(0)).to(self.device)
                loss = self.criterion(similarity_matrix, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss:.4f}")

            val_metrics = self.evaluate()
            print(f"Validation Metrics: {val_metrics}")

    def evaluate(self):
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                passage_input_ids = batch['passage_input_ids'].to(self.device)
                passage_attention_mask = batch['passage_attention_mask'].to(self.device)

                query_embeddings = self.model(query_input_ids, query_attention_mask)
                passage_embeddings = self.model(passage_input_ids, passage_attention_mask)

                similarity_scores = torch.matmul(query_embeddings, passage_embeddings.t())
                predictions = similarity_scores.argmax(dim=1)
                labels = torch.arange(similarity_scores.size(0)).to(self.device)

                total_correct += (predictions == labels).sum().item()
                total_samples += predictions.size(0)

        accuracy = total_correct / total_samples
        return {"accuracy": accuracy}