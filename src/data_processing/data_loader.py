import json
from typing import Dict, List
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_documents(self) -> Dict[str, List[Dict]]:
        """加载所有法律文档"""
        documents = {}
        doc_dir = self.data_dir / "documents"

        for doc_path in doc_dir.glob("*.json"):
            with open(doc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle single document or list of documents
                if isinstance(data, dict):
                    doc_id = str(data["DocumentID"])
                    if doc_id not in documents:
                        documents[doc_id] = []
                    documents[doc_id].append({
                        "ID": data["ID"],
                        "PassageID": data["PassageID"],
                        "text": data["Passage"],
                        "DocumentID": data["DocumentID"],

                    })
                elif isinstance(data, list):
                    for item in data:
                        doc_id = str(item["DocumentID"])
                        if doc_id not in documents:
                            documents[doc_id] = []
                        documents[doc_id].append({
                            "ID": item["ID"],  # 这里修改为item["ID"]
                            "PassageID": item["PassageID"],
                            "DocumentID": item["DocumentID"],
                            "text": item["Passage"]
                        })
        return documents

    def load_obliqa_dataset(self, split: str) -> List[Dict]:
        """加载ObliQA数据集"""
        file_path = self.data_dir / f"ObliQA_{split}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_unseen_questions(self) -> List[Dict]:
        """加载未见过的问题集"""
        file_path = self.data_dir / "RIRAG_Unseen_Questions.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)