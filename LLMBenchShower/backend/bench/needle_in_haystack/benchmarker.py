import os
import sys
import json
import random
import asyncio
import tiktoken
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchbase import BaseBench

# NeedleBench V2 Task Types
class TaskType:
    SINGLE_NEEDLE_RETRIEVAL = "single_needle_retrieval"
    MULTI_NEEDLE_RETRIEVAL = "multi_needle_retrieval" 
    MULTI_NEEDLE_REASONING = "multi_needle_reasoning"
    ANCESTRAL_TRACE_CHALLENGE = "ancestral_trace_challenge"

class NeedleInHaystackBenchmarker(BaseBench):
    def __init__(self):
        # Get the project root directory (go up 4 levels from backend/bench/needle_in_haystack)
        # From backend/bench/needle_in_haystack/benchmarker.py -> backend/bench/needle_in_haystack -> backend/bench -> backend -> LLMBenchShower -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        # Dataset paths using relative paths - datasets are in project_root/tests/test_data/
        self.english_dataset_dir = os.path.join(project_root, "tests", "test_data", "NeedleInHaystack", "PaulGrahamEssays")
        self.chinese_dataset_path = os.path.join(project_root, "tests", "test_data", "NeedleInHaystack", "Journey_to_the_West.txt")# ATC相关配置
        self.atc_names_file = None  # 暂时设为None，使用内置姓氏列表
        
        # Default needles for testing
        self.default_needles = {
            "English": [
                "The most important thing for a startup is to make something people want.",
                "Startups should focus on a small, specific market initially.",
                "The best way to get startup ideas is to look for problems you have yourself.",
                "Founders should be prepared for the emotional rollercoaster of building a company.",
                "The most successful startups often come from technical founders.",
                "Startups should aim for growth, not just profitability.",
                "The best time to start a startup is when you're young and have few responsibilities.",
                "Startups should avoid raising too much money too early.",
                "The most important quality in a founder is determination.",
                "Startups should focus on building a great product first, then worry about marketing."
            ],
            "Chinese": [
                "人生在世，应当有所作为。",
                "知足常乐是一种智慧。",
                "诚实守信是做人的根本。",
                "学而时习之，不亦说乎？",
                "千里之行，始于足下。",
                "书山有路勤为径，学海无涯苦作舟。",
                "天道酬勤，厚德载物。",
                "君子爱财，取之有道。",
                "三人行，必有我师焉。",
                "己所不欲，勿施于人。"
            ]
        }
        
        # Default retrieval questions
        self.default_questions = {
            "English": "Based on the provided text, what is the most important advice for startups?",
            "Chinese": "根据提供的文本，最重要的道理是什么？"
        }
        
        # ATC relationship templates and terms (from NeedleBench V2)
        self.relationship_templates_zh_CN = [
            '{A}是{B}的{relationship}。',
            '{B}的{relationship}是{A}。',
            '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
            '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
            '{A}在{B}的成长过程中，不仅仅是{B}的{relationship}，还是{B}的监护人。',
            '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
        ]
        
        self.relationship_templates_en = [
            "{A} is {B}'s {relationship}.",
            "{B}'s {relationship} is {A}.",
            ("{A}, as {B}'s {relationship}, "
             "has a significant impact on {B}'s upbringing."),
            ("{A} is not only {B}'s {relationship} "
             "but also {B}'s role model."),
            ("During {B}'s upbringing, {A} was not only {B}'s {relationship}, "
             "but also {B}'s guardian."),
            ('For {B}, {A} is not just a {relationship}, '
             'but also a friend.'),
            'For {B}, {A} is more than just a {relationship}; {A} is a lifelong mentor of {B}.',
        ]
        
        self.relationship_terms_zh_CN = [
            '父亲', '母亲', '爸爸', '妈妈', '爷爷', '奶奶', '姥姥', '姥爷', '外公', '外婆'
        ]
        
        self.relationship_terms_en = [
            'father', 'mother', 'dad', 'mom', 'grandfather', 'grandmother',
            'maternal grandmother', 'maternal grandfather', 'paternal grandfather', 'paternal grandmother'
        ]
        
        # Default test parameters
        self.default_params = {
            "context_length": 32768,
            "depth_percent": 50,
            "num_needles": 1,
            "language": "English",
            "task_type": TaskType.SINGLE_NEEDLE_RETRIEVAL
        }
    
    def _load_dataset(self, dataset_name: str) -> str:
        """Load the dataset content."""
        if dataset_name == "Journey_to_the_West":
            with open(self.chinese_dataset_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:  # PaulGrahamEssays
            content = ""
            essays_dir = Path(self.english_dataset_dir)
            for file_path in essays_dir.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content += f.read() + "\n\n"
            return content
    
    def _get_random_needle(self, language: str = "English") -> str:
        """Get a random needle from the default needles."""
        return random.choice(self.default_needles[language])
    
    def _get_needles_list(self, num_needles: int, language: str = "English") -> List[str]:
        """Get a list of needles for multi-needle testing."""
        needles_source = self.default_needles[language]
        if num_needles > len(needles_source):
            needles = needles_source * (num_needles // len(needles_source))
            needles.extend(needles_source[:num_needles % len(needles_source)])
            return needles
        else:
            return random.sample(needles_source, num_needles)
    
    def _insert_needle(self, context: str, needle: str, depth_percent: int) -> str:
        """Insert needle into context at specified depth percentage."""
        # Use tiktoken for token-level insertion
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize context and needle
        context_tokens = tokenizer.encode(context)
        needle_tokens = tokenizer.encode(needle)
        
        # Calculate insertion point
        insertion_point = int(len(context_tokens) * (depth_percent / 100))
        
        # Insert needle
        new_tokens = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
        
        # Decode back to text
        return tokenizer.decode(new_tokens)
    
    def _insert_multiple_needles(self, context: str, needles: List[str], depth_percent: int) -> str:
        """Insert multiple needles into context with equal spacing."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer.encode(context)
        
        # Calculate insertion points
        num_needles = len(needles)
        insertion_points = []
        for i in range(num_needles):
            point = int(len(context_tokens) * (depth_percent / 100) * (i + 1) / num_needles)
            insertion_points.append(point)
        
        # Insert needles in reverse order to maintain correct positions
        new_tokens = context_tokens
        for i, (needle, point) in enumerate(zip(reversed(needles), reversed(insertion_points))):
            needle_tokens = tokenizer.encode(needle)
            new_tokens = new_tokens[:point] + needle_tokens + new_tokens[point:]
        
        return tokenizer.decode(new_tokens)
    
    def _prepare_prompt(self, context: str, question: str, language: str = "English") -> str:
        """Prepare the prompt for the model."""
        if language == "Chinese":
            return f"""这是一个长文本能力测试。请仔细阅读下面的长文档，然后根据文档中的信息直接回答问题。

重要提示：请直接给出答案，不要添加额外的解释或分析。

长文档内容：

<文档>
{context}
</文档>

问题：{question}

请直接回答："""
        else:
            return f"""This is a long-text capability test. Please read the long document below carefully, then answer the question directly based on the information in the document.

Important: Please provide the answer directly without additional explanations or analysis.

Document content:

<Document>
{context}
</Document>

Question: {question}

Please answer directly: """
    
    def _evaluate_response(self, response: str, needle: str, language: str = "English") -> float:
        """Evaluate the model's response against the expected needle."""
        needle_clean = needle.strip().lower()
        response_clean = response.strip().lower()
        
        # 1. Clean response: remove quotes and markdown formatting
        response_clean = re.sub(r'[\*\*\_\`\"]', '', response_clean)  # Remove markdown formatting
        response_clean = re.sub(r'\\boxed\{[^}]*\}', '', response_clean)  # Remove boxed answers
        
        # 2. Extract direct answer (first sentence or key phrase)
        # Try to extract the most relevant part of the response
        sentences = re.split(r'[.!?]', response_clean)
        if sentences:
            # Use the first sentence as the main answer
            main_answer = sentences[0].strip()
            
            # Check if main answer contains needle
            if needle_clean in main_answer:
                return 1.0
            
            # Check word overlap in main answer
            needle_words = set(needle_clean.split())
            answer_words = set(main_answer.split())
            
            if len(needle_words) > 0:
                overlap = len(needle_words.intersection(answer_words)) / len(needle_words)
                
                # Enhanced scoring based on overlap
                if overlap >= 0.8:
                    return 0.9  # High semantic similarity
                elif overlap >= 0.6:
                    return 0.7  # Good semantic similarity
                elif overlap >= 0.4:
                    return 0.5  # Moderate semantic similarity
                elif overlap >= 0.2:
                    return 0.3  # Low semantic similarity
        
        # 3. Check full response for exact match
        if needle_clean in response_clean:
            return 0.8  # Found in full response but not in main answer
        
        # 4. Check for key phrases in full response
        key_phrases = [
            "startup", "important", "focus", "market", "growth", 
            "product", "founder", "company", "advice", "best"
        ] if language == "English" else [
            "重要", "道理", "智慧", "诚实", "学习", "行动", "勤奋", "道德"
        ]
        
        key_phrase_count = sum(1 for phrase in key_phrases if phrase in response_clean)
        if key_phrase_count >= 3:
            return 0.4  # Contains relevant key phrases
        elif key_phrase_count >= 1:
            return 0.2  # Contains some relevant phrases
        
        return 0.0  # No meaningful match
    
    def _evaluate_atc_response(self, response: str, expected_answer: str) -> float:
        """Evaluate ATC response using NeedleBench V2's evaluation logic."""
        # Extract answer from boxed format
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        
        if match:
            extracted_answer = match.group(1).strip().lower()
            expected_answer_clean = expected_answer.strip().lower()
            
            # Exact match
            if extracted_answer == expected_answer_clean:
                return 1.0
            
            # Check if expected answer is contained in extracted answer
            if expected_answer_clean in extracted_answer:
                return 1.0
                
            # Check if extracted answer is contained in expected answer
            if extracted_answer in expected_answer_clean:
                return 0.8
        
        return 0.0
    
    def _generate_atc_needles(self, num_needles: int, language: str) -> Dict:
        """Generate ATC needles for multi-needle reasoning using NeedleBench V2's power-of-2 distribution."""
        # Use built-in Chinese and English names
        chinese_names = ["张伟", "王芳", "李娜", "刘洋", "陈静", "杨明", "黄强", "赵丽", "周杰", "吴婷", 
                       "孙涛", "朱敏", "徐辉", "林静", "赵强", "王芳", "刘娜", "陈涛", "杨丽", "吴杰"]
        english_names = ["John", "Mary", "David", "Sarah", "Michael", "Emily", "James", "Emma", "Robert", "Olivia",
                       "William", "Ava", "Alexander", "Sophia", "Daniel", "Isabella", "Matthew", "Charlotte", "Ethan", "Amelia"]
        
        names = chinese_names if language == "Chinese" else english_names
        
        # Generate family relationship chain using power-of-2 distribution as in NeedleBench V2
        # Ensure we have enough names for the requested number of needles
        if num_needles > len(names):
            # If not enough names, repeat the list with suffixes
            repeated_names = []
            for i in range((num_needles // len(names)) + 1):
                for name in names:
                    repeated_names.append(f"{name}{i+1}" if i > 0 else name)
            names = repeated_names
        
        # NeedleBench V2 uses power-of-2 distribution: 2, 4, 8, 16, 32, 64, etc.
        # Generate a chain with exact number of needles requested
        selected_names = names[:num_needles]
        
        # Create relationship chain with proper NeedleBench V2 structure
        relationship_chain = []
        for i in range(len(selected_names) - 1):
            if language == "Chinese":
                # Use proper Chinese relationship terms (from youngest to oldest)
                relationships_zh = ["父亲", "母亲", "祖父", "祖母", "曾祖父", "曾祖母", "高祖父", "高祖母"]
                relationship_type = relationships_zh[i % len(relationships_zh)]
                relationship = f"{selected_names[i]}是{selected_names[i+1]}的{relationship_type}"
            else:
                # Use proper English relationship terms (from youngest to oldest)
                relationships_en = ["father", "mother", "grandfather", "grandmother", "great-grandfather", "great-grandmother", 
                                  "great-great-grandfather", "great-great-grandmother"]
                relationship_type = relationships_en[i % len(relationships_en)]
                relationship = f"{selected_names[i]} is the {relationship_type} of {selected_names[i+1]}"
            relationship_chain.append(relationship)
        
        # Generate retrieval question based on NeedleBench V2 format
        if language == "Chinese":
            retrieval_question = f"在上面提供的文本中，'{selected_names[-1]}'的能够向上追溯到的最年长的亲人是谁？请使用 \\boxed{{答案}} 的格式直接给出答案。"
        else:
            retrieval_question = f"Given the context described above, who is the eldest relative that '{selected_names[-1]}' can trace back to in the context? Please provide the answer directly using the format \\boxed{{answer}}."
        
        return {
            'needles': relationship_chain,
            'answer': selected_names[0],
            'retrieval_question': retrieval_question,
            'last_person': selected_names[-1]
        }
    
    async def _call_local_model(self, model, tokenizer, prompt: str) -> str:
        """Call a local model with the given prompt."""
        try:
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new text (after the prompt)
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
        except Exception as e:
            print(f"Error calling local model: {e}")
            return ""
    
    async def _call_api_model(self, client: Client, model: str, prompt: str) -> str:
        """Call an API model with the given prompt."""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling API model: {e}")
            return ""
    
    async def evaluate_local_llm(self, model, tokenizer, params: Dict = None) -> Dict:
        """Evaluate a local LLM using the needle-in-haystack benchmark."""
        if params is None:
            params = self.default_params
        
        # Get task type
        task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            return await self._evaluate_single_needle_retrieval(model, tokenizer, params)
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            return await self._evaluate_multi_needle_retrieval(model, tokenizer, params)
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            return await self._evaluate_multi_needle_reasoning(model, tokenizer, params)
        elif task_type == TaskType.ANCESTRAL_TRACE_CHALLENGE:
            return await self._evaluate_ancestral_trace_challenge(model, tokenizer, params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _evaluate_single_needle_retrieval(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate single needle retrieval task."""
        # Get parameters
        num_needles = params.get("num_needles", 1)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        if num_needles > 1:
            needles = self._get_needles_list(num_needles, language)
            context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
            needle_for_eval = needles[0]  # Use first needle for evaluation
        else:
            needle = self._get_random_needle(language)
            context_with_needles = self._insert_needle(context, needle, depth_percent)
            needle_for_eval = needle
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model.config._name_or_path,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "success": True,
            "task_type": "SINGLE_NEEDLE_RETRIEVAL"
        }
    
    async def _evaluate_multi_needle_retrieval(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate multi needle retrieval task."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        needles = self._get_needles_list(num_needles, language)
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        needle_for_eval = needles[0]  # Use first needle for evaluation
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model.config._name_or_path,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "needles": needles,
            "success": True,
            "task_type": "MULTI_NEEDLE_RETRIEVAL"
        }
    
    async def _evaluate_multi_needle_reasoning(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate multi needle reasoning task."""
        # Get parameters
        num_needles = params.get("num_needles", 4)  # 使用基于2的幂次
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles for reasoning - 使用NeedleBench V2的基于2的幂次分布
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model.config._name_or_path,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": "MULTI_NEEDLE_REASONING"
        }
    
    async def _evaluate_ancestral_trace_challenge(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate ancestral trace challenge (ATC) task."""
        # Get parameters
        num_needles = params.get("num_needles", 8)  # ATC使用基于2的幂次的稀疏分布
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model.config._name_or_path,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": "ANCESTRAL_TRACE_CHALLENGE"
        }
    
    async def evaluate_api_llm(self, client: Client, params: Dict = None) -> Dict:
        """Evaluate an API-based LLM using the needle-in-haystack benchmark."""
        if params is None:
            params = self.default_params
        
        # Get task type
        task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            return await self._evaluate_api_single_needle_retrieval(client, params)
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            return await self._evaluate_api_multi_needle_retrieval(client, params)
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            return await self._evaluate_api_multi_needle_reasoning(client, params)
        elif task_type == TaskType.ANCESTRAL_TRACE_CHALLENGE:
            return await self._evaluate_api_ancestral_trace_challenge(client, params)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def evaluate_needlebench_v2(self, client: Client = None, model=None, tokenizer=None, params: Dict = None) -> Dict:
        """
        Evaluate using NeedleBench V2's balanced scoring system.
        Overall score is the simple average of three main tasks:
        - Single Needle Retrieval
        - Multi Needle Retrieval  
        - Multi Needle Reasoning
        Each task gets equal weight.
        """
        if params is None:
            params = self.default_params
        
        # Store original task type
        original_task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        # Evaluate each task type
        task_results = {}
        
        # Single Needle Retrieval
        params["task_type"] = TaskType.SINGLE_NEEDLE_RETRIEVAL
        if client:
            task_results["single_needle_retrieval"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["single_needle_retrieval"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Multi Needle Retrieval
        params["task_type"] = TaskType.MULTI_NEEDLE_RETRIEVAL
        if client:
            task_results["multi_needle_retrieval"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["multi_needle_retrieval"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Multi Needle Reasoning
        params["task_type"] = TaskType.MULTI_NEEDLE_REASONING
        if client:
            task_results["multi_needle_reasoning"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["multi_needle_reasoning"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Calculate overall score (simple average of three tasks)
        scores = [result["score"] for result in task_results.values()]
        overall_score = sum(scores) / len(scores)
        
        # Restore original task type
        params["task_type"] = original_task_type
        
        return {
            "overall_score": overall_score,
            "task_scores": {
                "single_needle_retrieval": task_results["single_needle_retrieval"]["score"],
                "multi_needle_retrieval": task_results["multi_needle_retrieval"]["score"],
                "multi_needle_reasoning": task_results["multi_needle_reasoning"]["score"]
            },
            "task_results": task_results,
            "scoring_system": "NeedleBench V2 Balanced Scoring (equal weight average)"
        }
    
    async def _evaluate_api_single_needle_retrieval(self, client: Client, params: Dict) -> Dict:
        """Evaluate single needle retrieval task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 1)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        if num_needles > 1:
            needles = self._get_needles_list(num_needles, language)
            context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
            needle_for_eval = needles[0]  # Use first needle for evaluation
        else:
            needle = self._get_random_needle(language)
            context_with_needles = self._insert_needle(context, needle, depth_percent)
            needle_for_eval = needle
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "success": True,
            "task_type": "SINGLE_NEEDLE_RETRIEVAL"
        }
    
    async def _evaluate_api_multi_needle_retrieval(self, client: Client, params: Dict) -> Dict:
        """Evaluate multi needle retrieval task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        needles = self._get_needles_list(num_needles, language)
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        needle_for_eval = needles[0]  # Use first needle for evaluation
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "needles": needles,
            "success": True,
            "task_type": "MULTI_NEEDLE_RETRIEVAL"
        }
    
    async def _evaluate_api_multi_needle_reasoning(self, client: Client, params: Dict) -> Dict:
        """Evaluate multi needle reasoning task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles for reasoning
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": "MULTI_NEEDLE_REASONING"
        }
    
    async def _evaluate_api_ancestral_trace_challenge(self, client: Client, params: Dict) -> Dict:
        """Evaluate ancestral trace challenge (ATC) task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 5)  # ATC typically uses more needles
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": "ANCESTRAL_TRACE_CHALLENGE"
        }
    
    def evaluate_with_fallback(
        self,
        model_or_client: Any,
        model_name: str,
        subdataset_name: str = "PaulGrahamEssays",
        *args,
        **kwargs
    ) -> Dict:
        """Evaluate with automatic fallback from local model to API."""
        try:
            # Try local model evaluation first
            if hasattr(model_or_client, 'generate') and hasattr(model_or_client, 'config'):
                # This is likely a local model
                tokenizer = kwargs.get('tokenizer')
                if tokenizer is None:
                    raise ValueError("Tokenizer is required for local model evaluation")
                
                return self.evaluate_local_llm(model_or_client, tokenizer, subdataset_name, *args, **kwargs)
            
            # Fall back to API evaluation
            return self.evaluate_api_llm(model_or_client, model_name, subdataset_name, *args, **kwargs)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "model": model_name,
                "dataset": subdataset_name,
                "error": str(e),
                "success": False
            }

# Example usage
if __name__ == "__main__":
    # Create benchmarker
    benchmarker = NeedleInHaystackBenchmarker()
    
    # Example: Test with a simple prompt
    context = "This is a test context. " * 100
    needle = "The secret needle is here."
    question = "What is the secret needle?"
    
    context_with_needle = benchmarker._insert_needle(context, needle, 50)
    prompt = benchmarker._prepare_prompt(context_with_needle, question)
    
    print("Test prompt:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("\n" + "="*50 + "\n")
    
    # Test evaluation
    test_response = "The secret needle is here."
    score = benchmarker._evaluate_response(test_response, needle)
    print(f"Test evaluation score: {score}")
