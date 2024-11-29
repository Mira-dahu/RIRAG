from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 alpha: float = 0.5):
        self.dense_model = SentenceTransformer(model_name)
        self.bm25 = None
        self.passages = []
        self.passage_embeddings = None
        self.alpha = alpha

    def index_passages(self, passages: List[str]):
        self.passages = passages

        tokenized_passages = [p.split() for p in passages]
        self.bm25 = BM25Okapi(tokenized_passages)

        self.passage_embeddings = self.dense_model.encode(
            passages,
            convert_to_tensor=True,
            show_progress_bar=True
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_length = len(query.split())
        if query_length < 5:
            alpha = 0.7
        else:
            alpha = 0.3

        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        query_embedding = self.dense_model.encode(query, convert_to_tensor=True)
        dense_scores = np.array([
            float(self.passage_embeddings[i] @ query_embedding)
            for i in range(len(self.passages))
        ])
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))

        final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        return [(self.passages[i], final_scores[i]) for i in top_indices]