import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(self, model_name="microsoft/mdeberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.pooler_output[:, 0]  # 取[CLS]标记的输出作为相似度分数
        return logits