"""Evaluation metrics for LongBench tasks.

This module provides various metrics for evaluating long-context LLM performance
on different types of tasks including QA, retrieval, summarization, etc.

Based on: https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
"""

import re
import string
from typing import List
from collections import Counter

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def normalize_answer(s: str) -> str:
    """Normalize English answers for comparison.
    
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Normalize Chinese answers for comparison.
    
    Lower text and remove punctuation, extra whitespace.
    """

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score based on counting numbers in prediction.
    
    Counts how many numbers in prediction match the ground truth number.
    """
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score for passage retrieval tasks (English).
    
    Extracts paragraph numbers from prediction and ground truth,
    checking for matches.
    """
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    
    if not matches:
        return 0.0
    
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score for passage retrieval tasks (Chinese).
    
    Extracts paragraph numbers from prediction and ground truth,
    checking for matches.
    """
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    
    if not matches:
        return 0.0
    
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score for code completion tasks using fuzzy matching.
    
    Extracts actual code lines and compares using fuzzy string matching.
    """
    if not FUZZYWUZZY_AVAILABLE:
        # Fallback: simple exact match
        return 1.0 if prediction.strip() == ground_truth.strip() else 0.0
    
    all_lines = prediction.lstrip('\n').split('\n')
    prediction_code = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction_code = line
            break
    
    return float(fuzz.ratio(prediction_code, ground_truth) / 100)


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Score for classification tasks.
    
    Checks if the ground truth class appears in the prediction.
    Handles cases where multiple classes might appear in prediction.
    """
    em_match_list = []
    all_classes = kwargs.get("all_classes", [])
    
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    
    # Remove matches that are substrings of other matches
    for match_term in em_match_list[:]:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    
    return float(score)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Calculate ROUGE-L score for summarization tasks (English).
    
    Uses the ROUGE metric to compare prediction with ground truth.
    """
    if not ROUGE_AVAILABLE:
        # Fallback: simple F1-like score
        return simple_similarity(prediction, ground_truth)
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return float(scores["rouge-l"]["f"])
    except Exception as e:
        # Return 0 if ROUGE calculation fails
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Calculate ROUGE-L score for summarization tasks (Chinese).
    
    Tokenizes Chinese text using jieba before calculating ROUGE.
    """
    if not JIEBA_AVAILABLE:
        # Fallback: character-level comparison
        return simple_similarity(prediction, ground_truth)
    
    try:
        prediction_tokens = " ".join(list(jieba.cut(prediction, cut_all=False)))
        ground_truth_tokens = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
        return rouge_score(prediction_tokens, ground_truth_tokens)
    except Exception as e:
        return 0.0


def simple_similarity(text1: str, text2: str) -> float:
    """Simple similarity score based on character overlap.
    
    Fallback when more sophisticated metrics are not available.
    """
    if not text1 or not text2:
        return 0.0
    
    common = sum(1 for c in text1 if c in text2)
    return float(common) / max(len(text1), len(text2))


def f1_score(tokens1: List[str], tokens2: List[str], **kwargs) -> float:
    """Calculate F1 score based on token overlap.
    
    Compares two token lists and calculates precision, recall, and F1.
    """
    common = Counter(tokens1) & Counter(tokens2)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(tokens1) if tokens1 else 0.0
    recall = 1.0 * num_same / len(tokens2) if tokens2 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """F1 score for English QA tasks.
    
    Normalizes both strings, tokenizes, and calculates F1.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """F1 score for Chinese QA tasks.
    
    Uses jieba for tokenization and calculates F1 on tokens.
    """
    if not JIEBA_AVAILABLE:
        # Fallback: character-level F1
        prediction_chars = list(prediction)
        ground_truth_chars = list(ground_truth)
        return f1_score(prediction_chars, ground_truth_chars)
    
    try:
        prediction_tokens = list(jieba.cut(prediction, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        
        prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        
        prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
        ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
        
        return f1_score(prediction_tokens, ground_truth_tokens)
    except Exception as e:
        return 0.0


# Mapping from task names to scoring functions
TASK_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def get_metric_function(task_name: str):
    """Get the appropriate metric function for a task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        The corresponding metric function, or qa_f1_score as default
    """
    return TASK_METRICS.get(task_name, qa_f1_score)
