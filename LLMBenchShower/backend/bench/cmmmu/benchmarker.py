import os
import re
import json
import string
import torch
import ast
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path
from PIL import Image

class CMMMUBenchmarker(BaseBench):
    """
    Benchmarker for CMMMU dataset (Text-Only Version).
    Integrated with robust evaluation logic similar to official CMMMU scripts.
    """
    def __init__(self):
        self.dataset_path = get_dataset_path("CMMMU")

    def _load_dataset(self, subdataset_name: str) -> List[Dict]:
        """Load dataset from local JSONL files."""
        # dataset_path/subdataset_name/subdataset_name.jsonl
        subdir = os.path.join(self.dataset_path, subdataset_name)
        file_path = os.path.join(subdir, f"{subdataset_name}.jsonl")
        self.image_path = subdir
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item['subject_name'] = subdataset_name
                    data.append(item)
            return data

        except Exception as e:
            raise RuntimeError(f"Error loading dataset {subdataset_name}: {e}")

    def _clean_text(self, text: str) -> str:
        """Remove <img...> tags and extra whitespace."""
        if not text:
            return ""
        text = re.sub(r'<img="[^"]+">', '', str(text))
        return text.strip()
    
    def _process_question_with_image(self, question: str) -> Dict:
        """
        Args:
            question (str):original text contains <img="([^"]+)"。
        Returns:
            Dict: {"text": text,"images": image}
        """
        # image
        match = re.search(r'<img="([^"]+)"', question)
        image_paths = self.image_path + "/"+match.group(1)
        image = Image.open(image_paths).convert("RGB")
        # text
        text = re.sub(r'<img[^>]+>', '', question).strip()

        return {
            "text": text,
            "images": image
        }
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for fuzzy matching (used in Fill-in-the-blank).
        Removes punctuation, lowers case, strips whitespace.
        """
        if not text:
            return ""
        text = str(text).lower()
        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # remove extra spaces
        text = " ".join(text.split())
        return text

    def _prepare_prompt(self, item: Dict) -> str:
        """Prepare the prompt from a dataset item (MCQ format).

        Args:
            item (Dict): A data item containing question and options.

        Returns:
            str: The formatted prompt.
        """
        q_type = item.get("type")
        
        if q_type == "选择":
            context = "[选择题]:"
            question = item.get("question", "")
            if "<img" in question:
                question = self._process_question_with_image(question)
            opt_a = item.get("option1", "")
            opt_b = item.get("option2", "")
            opt_c = item.get("option3", "")
            opt_d = item.get("option4", "")
            # 选择题prompt模板
            prompt = (
                f"\n\nQuestion:{context}{question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\nAnswer:"
            )
        elif q_type == "判断":
            context = "[判断题]:"
            question = item.get("question", "")
            if "<img" in question:
                question = self._process_question_with_image(question)
            # 判断题prompt模板
            prompt = (
                f"\n\nQuestion:{context}{question}\n\nAnswer:"
            )
        elif q_type == "填空":
            context = "[填空题]:"
            question = item.get("question", "")
            if "<img" in question:
                question = self._process_question_with_image(question)
            # 填空题prompt模板
            prompt = (
                f"\n\nQuestion:{context}{question}\n\nAnswer:"
            )
        else:
            question = item.get("question", "")
            if "<img" in question:
                question = self._process_question_with_image(question)
            prompt = (
                f"\n\nQuestion:{question}\n\nAnswer:"
            )
        print("[debug] prompt prepared:", prompt)
        return prompt

    def _extract_prediction_label(self, prediction: str, item: Dict) -> str:
        """
        Robust answer extraction based on question type.
        1. For MCQ (选择题), look for explicit letter (A/B/C/D) or match option content.
        2. For 判断题, look for "正确/错误" or "True/False".
        3. For 填空题, return the normalized prediction directly.
        """
        pred = prediction.strip()
        if not pred:
            return ""

        q_type = item.get("type", "")

        # 1. 选择题 (MCQ)
        if q_type == "选择":
            # 优先提取明确的选项字母
            match = re.search(r"(?:答案|选|Answer|Choice|^)[\s:：]*([ABCD])(?![a-zA-Z])", pred, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # 尝试匹配选项内容
            pred_normalized = self._normalize_text(pred)
            chars = "ABCD"
            for i, char in enumerate(chars):
                opt_text = ""
                if f"option{i+1}" in item:
                    opt_text = str(item[f"option{i+1}"])
                elif "options" in item:
                    opts = item["options"]
                    if isinstance(opts, str):
                        try: opts = ast.literal_eval(opts)
                        except: opts = []
                    if isinstance(opts, list) and i < len(opts):
                        opt_text = str(opts[i])
                if opt_text:
                    opt_clean = self._normalize_text(self._clean_text(opt_text))
                    if opt_clean and opt_clean == pred_normalized:
                        return char  # 找到了对应的选项字母

            # 兜底：寻找最后一个出现的字母
            matches = re.findall(r"([ABCD])", pred.upper())
            if matches:
                return matches[-1]

        # 2. 判断题 (True/False)
        elif q_type == "判断":
            # 尝试匹配 "正确/错误" 或 "True/False"
            if re.search(r"(正确|True)", pred, re.IGNORECASE):
                return "正确"
            elif re.search(r"(错误|False)", pred, re.IGNORECASE):
                return "错误"

        # 3. 填空题 (Fill-in-the-blank)
        elif q_type == "填空":
            # 返回标准化后的预测内容
            return self._normalize_text(pred)

        # 4. 其他类型
        return ""

    def _calculate_score(self, prediction: str, item: Dict, ground_truth: str) -> float:
        """Calculate score with type-specific logic."""
        q_type = item.get("type", "选择")

        if q_type == "选择":
            extracted_pred = self._extract_prediction_label(prediction, item)
            if extracted_pred == ground_truth.upper():
                return 1.0
        else:
            clean_pred = self._normalize_text(prediction)
            clean_gt = self._normalize_text(ground_truth)
            # 1. 完全匹配
            if clean_pred == clean_gt:
                return 1.0
            # 2. 包含匹配
            if clean_gt in clean_pred and len(clean_gt) > 0:
                return 1.0
                
        return 0.0

    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM on CMMMU dataset."""
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "local",
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
                "score": 0.0,
            }
        }
        
        all_scores = []
        
        for item in dataset:
            try:
                prompt = self._prepare_prompt(item)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9
                    )
                #print("[debug local] outputs:", outputs)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                #print("[debug local] response1:", response)
                # Remove prompt
                response = response[:].strip()
                #print("[debug local] response2:", response)
                ground_truth = item.get("answer", "")
                
                # Calculate score
                score = self._calculate_score(response, item, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": response, 
                    "extracted_answer": self._extract_prediction_label(response, item),
                    "ground_truth": ground_truth,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
                
            except Exception as e:
                print(f"Error processing item: {e}")
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": item.get("answer", ""),
                    "score": 0.0,
                })
        
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results

    def evaluate_api_llm(
        self,
        client: Client,
        model: str,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate an API LLM on CMMMU dataset."""
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "api",
            "model": model,
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
                "score": 0.0,
            }
        }
        
        all_scores = []
        
        for item in dataset:
            try:
                prompt = self._prepare_prompt(item)
                
                response = client.messages.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=512,
                    temperature=0.7,
                )
                #print("[debug api] response:", response)
                prediction = response.choices[0].message.content.strip()
                ground_truth = item.get("answer", "")
                
                score = self._calculate_score(prediction, item, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": prediction,
                    "extracted_answer": self._extract_prediction_label(prediction, item),
                    "ground_truth": ground_truth,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
                
            except Exception as e:
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": item.get("answer", ""),
                    "score": 0.0,
                })
        
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results