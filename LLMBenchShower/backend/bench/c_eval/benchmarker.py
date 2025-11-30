import os
import torch
import re
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path, get_sub_datasets

class C_EvalBenchmarker(BaseBench):
    """Benchmarker for CEval dataset.
    
    C-Eval is a comprehensive Chinese evaluation suite for foundation models.
    It consists of multiple-choice questions across various disciplines.
    Reference: https://huggingface.co/datasets/ceval/ceval-exam
    """
    def __init__(self):
        """Initialize the CEval benchmarker."""
        self.dataset_path = get_dataset_path("C-Eval")
        self.sub_datasets = get_sub_datasets("C-Eval")

    def _load_dataset(self, subdataset_name: str) -> List[Dict]:
        """Load a subdataset using Hugging Face datasets.

        Args:
            subdataset_name (str): The name of the subdataset (subject) to load.

        Returns:
            List[Dict]: The loaded dataset as a list of dictionaries.
        """
        # 加载具体子科目 (Hugging Face 直接加载)
        try:
            # path: 远程仓库地址
            # name: 具体科目 (e.g.'computer_network')
            # split: C-Eval通常验证用 'val' (test集无答案)
            # cache_dir: 指定本地缓存目录
            dataset = load_dataset(
                path="ceval/ceval-exam",
                name=subdataset_name,
                split="val",
                trust_remote_code=True,
                cache_dir=self.dataset_path
            )
            data_list = []
            for item in dataset:
                item_dict = dict(item)
                item_dict['subject_name'] = subdataset_name
                data_list.append(item_dict)
            #print("[debug] data_list loaded:", data_list)
            return data_list
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {subdataset_name} via Hugging Face: {e}")

    def _prepare_prompt(self, item: Dict) -> str:
        """Prepare the prompt from a dataset item (MCQ format).

        Args:
            item (Dict): A data item containing question and options.

        Returns:
            str: The formatted prompt.
        """
        question = item.get("question", "")
        opt_a = item.get("A", "")
        opt_b = item.get("B", "")
        opt_c = item.get("C", "")
        opt_d = item.get("D", "")
        
        # 简单的选择题prompt模板
        prompt = (
            f"\n\nQuestion:{question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\nAnswer:"
        )
        print("[debug] prompt prepared:", prompt)
        return prompt

    def _extract_prediction_label(self, prediction: str) -> str:
        """Extract the option letter (A/B/C/D) from model output.
        
        Args:
            prediction (str): The raw model output.
            
        Returns:
            str: The extracted letter or empty string.
        """
        # 清理空白字符
        pred = prediction.strip()
        if not pred:
            return ""
            
        # 1. 如果第一个字符就是选项
        if pred[0].upper() in ["A", "B", "C", "D"]:
            return pred[0].upper()
            
        # 2. 正则
        match = re.search(r"(?:答案|选|Choice|Answer)[\s:：]*([ABCD])", pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 3. 如果没找到明确标记，尝试找最后一个出现的选项字母
        matches = re.findall(r"([ABCD])", pred.upper())
        if matches:
            return matches[-1]
            
        return ""

    def _calculate_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate score (0 or 1) for MCQ.

        Args:
            prediction (str): The model's raw prediction text.
            ground_truth (str): The correct option letter (e.g., "A").

        Returns:
            float: 1.0 if correct, 0.0 otherwise.
        """
        extracted_pred = self._extract_prediction_label(prediction)
        
        # 比较提取出的答案和标准答案
        if extracted_pred == ground_truth.strip().upper():
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
        """Evaluate a local LLM on CEval dataset."""
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
                score = self._calculate_score(response, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": response, 
                    "extracted_answer": self._extract_prediction_label(response),
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
        """Evaluate an API LLM on CEval dataset."""
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
                
                score = self._calculate_score(prediction, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": prediction,
                    "extracted_answer": self._extract_prediction_label(prediction),
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