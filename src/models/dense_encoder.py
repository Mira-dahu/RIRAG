import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DenseEncoder(nn.Module):
    def __init__(self, model_name="bert-base-chinese", pooling="mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                            min=1e-9)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]

        return embeddings