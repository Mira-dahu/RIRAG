import json
import requests
import pandas as pd
from typing import List, Dict

class AnswerGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_base = "https://api.deepseek.com/v1/chat/completions"

    def generate_answer(self, question: str, relevant_passages: List[str], documents: Dict[str, Dict],
                        question_id: str) -> str:
        df = pd.read_excel('data/raw/adjust1.xlsx')
        try:
            long_value = df.loc[df['QuestionID'] == question_id, 'long'].values[0]
        except (KeyError, IndexError):
            long_value = 0

        max_tokens = 800 if long_value == 1 else 1000

        prompt = self._construct_prompt(question, relevant_passages, documents)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a professional ADGM legal advisor.  

Please follow these guidelines when answering questions:  
1. Base your answers solely on the provided legal documents  
2. Cite specific legal provisions and passages accurately  
3. Use professional yet accessible language  
4. Ensure logical and complete responses  
5. Explicitly state when information is not found in the provided documents  

Recommended response structure:  
1. Direct answer to the core question  
2. Citations of supporting legal provisions  
3. Additional explanations where necessary  
4. Special notes or cautions if applicable"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _construct_prompt(self, question: str, passages: List[str], documents: Dict[str, List[Dict]]) -> str:
        context = []

        for i, passage in enumerate(passages):
            doc_info = None
            # 遍历 documents 的每个文档
            for doc_key, doc_passages in documents.items():
                # 遍历每个文档中的段落
                for info in doc_passages:
                    if info['text'] == passage:  # 假设这是需要匹配的逻辑
                        doc_info = info
                        break
                if doc_info:
                    break  # 如果找到了匹配的段落，就跳出循环

            if doc_info:
                context.append(
                    f"Reference {i + 1} (DocumentID: {doc_info['DocumentID']}, PassageID: {doc_info['PassageID']}):\n{passage}\n\n"
                )
            else:
                context.append(f"Reference {i + 1}: Passage not found in documents.\n\n")

        return f"""Question: {question}  

Relevant Legal Documents:  

{''.join(context)}  

Based on the above legal documents, please provide a professional, accurate, and easy-to-understand answer. Ensure to:  
1. Directly address the key points of the question  
2. Cite relevant legal provisions using the format (DocumentID: X, PassageID: Y)  
3. Provide additional explanations when necessary  
4. Follow the recommended response structure  
5. Clearly indicate if any required information is not present in the documents.  
"""