import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Tuple

class CrossEncoder(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large"):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.gradient_checkpointing = True
        
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits).squeeze(-1)

    def encode_pair(self, question: str, passage: str) -> dict:
        input_text = f"Question: {question} [SEP] Passage: {passage}"
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding

def rerank_passages(
    cross_encoder: CrossEncoder,
    question: str,
    retrieved_passages: List[Tuple[str, float]],
    batch_size: int = 4  # 由于large模型显存占用更大，减小batch size
) -> List[Tuple[str, float]]:
    cross_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_encoder = cross_encoder.to(device)
    
    all_scores = []
    
    for i in range(0, len(retrieved_passages), batch_size):
        batch_passages = retrieved_passages[i:i + batch_size]
        batch_inputs = []
        
        for passage, _ in batch_passages:
            encoding = cross_encoder.encode_pair(question, passage)
            batch_inputs.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
            
        with torch.no_grad(), torch.cuda.amp.autocast():  # 俺们用混合精度加速
            batch_input_ids = torch.cat([x['input_ids'] for x in batch_inputs]).to(device)
            batch_attention_mask = torch.cat([x['attention_mask'] for x in batch_inputs]).to(device)
            
            batch_scores = cross_encoder(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            all_scores.extend(batch_scores.cpu().tolist())
    
    ranked_results = []
    for (passage, orig_score), rerank_score in zip(retrieved_passages, all_scores):
        final_score = 0.3 * orig_score + 0.7 * rerank_score
        ranked_results.append((passage, final_score))
    
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results
