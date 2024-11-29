import unittest
from src.retrieval.bm25_retriever import BM25Retriever

class TestRetrieval(unittest.TestCase):
    def test_retrieve(self):
        passages = ["This is a test passage.", "Another test passage here."]
        retriever = BM25Retriever(passages)
        results = retriever.retrieve("test", top_k=1)
        self.assertEqual(len(results), 1)