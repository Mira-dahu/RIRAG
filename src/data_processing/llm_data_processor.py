import pandas as pd
from typing import List, Dict, Tuple
import json
import os

import pandas as pd
from typing import List, Dict, Tuple
import json


class LLMDataProcessor:
    def __init__(self):
        self.evaluation_criteria = {
            "信息不足": "要点≤3个",
            "信息适量": "要点4-5个"
        }

    def load_evaluation_data(self, excel_path: str) -> pd.DataFrame:
        """加载Excel评估数据"""
        try:
            print(f"Attempting to load Excel file from: {excel_path}")
            print(f"Current working directory: {os.getcwd()}")

            if not os.path.exists(excel_path):
                print(f"Error: File not found at {excel_path}")
                return None

            df = pd.read_excel(excel_path)
            print(f"Successfully loaded Excel file. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            return None

    def prepare_training_samples(self, df: pd.DataFrame) -> List[Dict]:
        """准备LLM微调的训练样本"""
        if df is None:
            print("DataFrame is None, cannot prepare training samples")
            return []

        training_samples = []
        print("Starting to prepare training samples...")

        for idx, row in df.iterrows():
            try:
                # 解析检索到的段落
                passages = row['RetrievedPassage(s)'].split('\n') if pd.notna(row['RetrievedPassage(s)']) else []

                # 构建训练样本
                sample = {
                    "question": row['Question'],
                    "context": passages,
                    "answer": row['answer'],
                    "evaluation": {
                        "insufficient_info": bool(row['信息不足(要点≤3个)']),
                        "adequate_info": bool(row['信息适量(要点4-5个)'])
                    }
                }
                training_samples.append(sample)

                if idx % 100 == 0:
                    print(f"Processed {idx} samples...")

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue

        print(f"Completed processing {len(training_samples)} training samples")
        return training_samples

    def create_prompt_template(self, sample: Dict) -> str:
        """创建提示模板"""
        context = "\n".join([f"段落{i + 1}: {p}" for i, p in enumerate(sample['context'])])

        prompt = f"""问题: {sample['question']}

相关法律文档:
{context}

Please provide a professional, accurate, and easy-to-understand answer based on the above legal documents. Requirements:
1. Directly address the key points of the question.
2. Cite relevant legal provisions.
3. Ensure the number of answer points is moderate (4-5).
4. Provide additional explanations if necessary.
5. If the required information is missing in the document, clearly indicate it.

Reference answer quality standard:
{sample['answer']}

Please generate your answer:"""

        return prompt