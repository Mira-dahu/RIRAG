def recall_at_k(retrieved, relevant, k=10):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)