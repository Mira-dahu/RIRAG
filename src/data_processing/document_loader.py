import json
import os

def load_documents(dir_path):
    documents = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.extend(json.load(f))
    return documents

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions