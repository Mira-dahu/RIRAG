from.bm25_retriever import BM25Retriever
from.dense_retriever import DenseRetriever

def select_passages(query, passages, method='bm25', top_k=10):
    retriever = BM25Retriever(passages) if method == 'bm25' else DenseRetriever(passages)
    return retriever.retrieve(query, top_k)