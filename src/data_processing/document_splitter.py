import json

def split_documents(file_path):
    with open(file_path, 'r') as f:
        documents = json.load(f)
    passages = []
    for doc in documents:
        passages.extend(doc['passages'])  # Assumes documents are in a list and have a 'passages' field
    return passages