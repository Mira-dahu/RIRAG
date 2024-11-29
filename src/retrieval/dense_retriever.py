from sentence_transformers import SentenceTransformer, util

class DenseRetriever:
    def __init__(self, passages):
        self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        self.corpus_embeddings = self.model.encode(passages, convert_to_tensor=True)

    def retrieve(self, query, top_k=10):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.dot_score(query_embedding, self.corpus_embeddings)[0].cpu().numpy()
        top_k_indices = scores.argsort()[-top_k:][::-1]
        return [passages[i] for i in top_k_indices]