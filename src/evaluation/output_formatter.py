import json
from typing import List, Dict


class OutputFormatter:
    def __init__(self, system_name: str = "LegalTriever"):
        self.system_name = system_name

    def save_trec_format(self, results: List[Dict], output_path: str):
        """保存 TREC 格式结果"""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                question_id = result["QuestionID"]
                for rank, (ID, score) in enumerate(result["Rankings"], 1):
                    line = f"{question_id} 0 {ID} {rank} {score:.4f} {self.system_name}\n"
                    f.write(line)

    def save_json_format(self, results: List[Dict], output_path: str):
        """保存JSON格式结果"""
        output_data = []
        for result in results:
            # 构建RetrievedPassage(s)字典
            retrieved_passages = {}
            for passage_info in result["PassageInfo"]:
                passage_id = passage_info["ID"]
                retrieved_passages[passage_id] = {
                    "DocumentID": passage_info["DocumentID"],
                    "PassageID": passage_info["PassageID"],
                    "Passage": passage_info["PassageText"]
                }

            output_item = {
                "QuestionID": result["QuestionID"],
                "Question": result["Question"],
                "RetrievedPassage(s)": retrieved_passages,
                "answer": result["Answer"]
            }
            output_data.append(output_item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

