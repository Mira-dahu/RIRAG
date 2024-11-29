
import rank_bm25
import nltk
import nltk
class BM25Retriever:
    def __init__(self, passages):
        self.passages = passages
        tokenized_corpus = [nltk.word_tokenize(passage) for passage in passages]
        self.bm25 = rank_bm25(tokenized_corpus)

    def retrieve(self, query, top_k=10):
        tokenized_query = nltk.word_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.passages[i] for i in top_k_indices]