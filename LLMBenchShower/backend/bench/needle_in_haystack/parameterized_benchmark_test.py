#!/usr/bin/env python3
"""
参数化基准测试框架
支持多维度参数组合测试，全面评估模型在不同条件下的性能
"""

import os
import sys
import asyncio
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加父目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bench.needle_in_haystack.benchmarker import NeedleInHaystackBenchmarker, TaskType

class ParameterizedBenchmarkTester:
    def __init__(self):
        self.benchmarker = NeedleInHaystackBenchmarker()
        self.local_model = None
        self.local_tokenizer = None
        self.api_client = None
        self.results = []
        
        # 定义参数空间
        self.parameter_space = {
            # 上下文长度分组 (tokens) - 按模型能力级别分组
            "context_length_groups": {
                "short": [4096, 8192],          # 短文本 - 基础模型能力
                "medium": [16384, 32768],        # 中等文本 - 标准模型能力
                "long": [65536, 128000],         # 长文本 - 高级模型能力
                "extra_long": [256000, 512000],  # 超长文本 - 顶级模型能力
                "extreme": [1000000]             # 极长文本 - 前沿模型能力
            },
            
            # 埋针深度百分比 (0-100) - 更细粒度的深度分布
            "depth_percents": [
                0,   # 开始位置
                10,  # 开头附近
                21,  # 前20%处
                31,  # 前30%处
                42,  # 前40%处
                52,  # 中间位置
                63,  # 后40%处
                73,  # 后30%处
                84,  # 后20%处
                94,  # 结尾附近
                100  # 结束位置
            ],
            
            # 针数配置 - 基于NeedleBench V2的2的幂次稀疏分布
            "needle_configs": {
                TaskType.SINGLE_NEEDLE_RETRIEVAL: [1],
                TaskType.MULTI_NEEDLE_RETRIEVAL: [2, 4, 8],  # 2^1, 2^2, 2^3
                TaskType.MULTI_NEEDLE_REASONING: [4, 8, 16],  # 2^2, 2^3, 2^4
                TaskType.ANCESTRAL_TRACE_CHALLENGE: [8, 16, 32]  # 2^3, 2^4, 2^5 - 基于2的幂次的稀疏分布
            },
            
            # 语言配置及对应长度缓冲
            "language_configs": {
                "English": {
                    "length_buffer": 3000,  # 英文需要更大的缓冲
                    "guide": True
                },
                "Chinese": {
                    "length_buffer": 200,   # 中文需要较小的缓冲
                    "guide": True
                }
            },
            
            # 任务类型
            "task_types": [
                TaskType.SINGLE_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_REASONING,
                TaskType.ANCESTRAL_TRACE_CHALLENGE
            ]
        }
    
    async def setup_local_model(self, model_path: str):
        """设置本地模型"""
        try:
            print(f"加载本地模型: {model_path}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("本地模型加载成功")
            return True
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            return False
    
    def setup_api_client(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """设置DeepSeek API客户端"""
        try:
            print("设置DeepSeek API客户端")
            self.api_client = Client(
                api_key=api_key,
                base_url=base_url
            )
            print("API客户端设置成功")
            return True
        except Exception as e:
            print(f"API客户端设置失败: {e}")
            return False
    
    def generate_test_combinations(self, max_combinations: int = 100, length_group: str = None):
        """生成测试参数组合
        
        Args:
            max_combinations: 最大测试组合数
            length_group: 上下文长度组 (short/medium/long/extra_long/extreme)，None表示所有组
        """
        combinations = []
        
        # 获取上下文长度
        context_lengths = []
        if length_group and length_group in self.parameter_space["context_length_groups"]:
            # 只使用指定组的上下文长度
            context_lengths = self.parameter_space["context_length_groups"][length_group]
        else:
            # 使用所有组的上下文长度
            for group_lengths in self.parameter_space["context_length_groups"].values():
                context_lengths.extend(group_lengths)
        
        # 计算每个任务类型的最大测试数，确保均衡分布
        task_types = self.parameter_space["task_types"]
        tests_per_task = max_combinations // len(task_types)
        remaining_tests = max_combinations % len(task_types)
        
        # 为每个任务类型生成测试组合
        for i, task_type in enumerate(task_types):
            task_combinations = []
            num_needles_options = self.parameter_space["needle_configs"][task_type]
            
            # 计算该任务类型的测试数
            task_max_tests = tests_per_task
            if i < remaining_tests:
                task_max_tests += 1
            
            # 为每个针数配置生成测试
            for num_needles in num_needles_options:
                # 为每个语言配置生成测试
                for language, lang_config in self.parameter_space["language_configs"].items():
                    # 为每个上下文长度生成测试
                    for context_length in context_lengths:
                        # 智能选择深度点（避免过多组合）
                        selected_depths = self._select_optimal_depths(task_type, num_needles)
                        
                        for depth_percent in selected_depths:
                            combination = {
                                "context_length": context_length,
                                "depth_percent": depth_percent,
                                "num_needles": num_needles,
                                "language": language,
                                "task_type": task_type,
                                "model_name": "deepseek-chat",
                                "length_buffer": lang_config["length_buffer"],
                                "guide": lang_config["guide"]
                            }
                            
                            task_combinations.append(combination)
                            
                            # 限制该任务类型的组合数量
                            if len(task_combinations) >= task_max_tests:
                                break
                        if len(task_combinations) >= task_max_tests:
                            break
                    if len(task_combinations) >= task_max_tests:
                        break
                if len(task_combinations) >= task_max_tests:
                    break
            
            # 如果该任务类型的组合数不足，添加一些关键测试
            if len(task_combinations) < task_max_tests:
                self._add_critical_combinations(task_combinations, task_type, context_lengths, task_max_tests)
            
            combinations.extend(task_combinations)
        
        # 如果总数超过限制，截断
        return combinations[:max_combinations]
    
    def _select_optimal_depths(self, task_type, num_needles, max_depths=5):
        """智能选择最优深度点，避免过多冗余测试
        
        根据任务类型和针数选择最具代表性的深度点
        """
        all_depths = self.parameter_space["depth_percents"]
        
        # 对于不同任务类型和针数，选择不同的关键深度点
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            # 单针检索：重点测试极端位置和中间位置
            key_depths = [0, 52, 100, 10, 94]
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            # 多针检索：测试均匀分布的深度
            step = len(all_depths) // max_depths
            key_depths = all_depths[::step][:max_depths]
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            # 多针推理：重点测试中间区域和后半部分
            key_depths = [52, 63, 73, 84, 94]
        else:  # ANCESTRAL_TRACE_CHALLENGE
            # 祖源追溯：重点测试前半部分和中间区域
            key_depths = [0, 21, 31, 42, 52]
        
        # 确保选择的深度点在有效范围内
        selected_depths = []
        for depth in key_depths:
            if depth in all_depths and depth not in selected_depths:
                selected_depths.append(depth)
                if len(selected_depths) >= max_depths:
                    break
        
        return selected_depths
    
    def _add_critical_combinations(self, combinations, task_type, context_lengths, max_tests):
        """添加关键测试组合，确保覆盖最重要的参数组合
        """
        critical_combinations = []
        num_needles_options = self.parameter_space["needle_configs"][task_type]
        
        # 添加关键组合：中间深度，不同上下文长度
        for language, lang_config in self.parameter_space["language_configs"].items():
            for num_needles in num_needles_options:
                for context_length in sorted(set(context_lengths))[:3]:  # 取前3个不同的长度
                    critical_combination = {
                        "context_length": context_length,
                        "depth_percent": 52,  # 中间位置
                        "num_needles": num_needles,
                        "language": language,
                        "task_type": task_type,
                        "model_name": "deepseek-chat",
                        "length_buffer": lang_config["length_buffer"],
                        "guide": lang_config["guide"]
                    }
                    
                    if critical_combination not in combinations and critical_combination not in critical_combinations:
                        critical_combinations.append(critical_combination)
                        if len(combinations) + len(critical_combinations) >= max_tests:
                            break
                if len(combinations) + len(critical_combinations) >= max_tests:
                    break
            if len(combinations) + len(critical_combinations) >= max_tests:
                break
        
        # 添加到主组合列表
        combinations.extend(critical_combinations[:max_tests - len(combinations)])
    
    async def run_single_test(self, params: Dict, use_local: bool = True):
        """运行单个测试"""
        try:
            # 根据语言配置添加额外参数
            language = params.get("language", "English")
            lang_config = self.parameter_space["language_configs"].get(language, {})
            
            # 合并语言特定参数
            test_params = params.copy()
            test_params.update({
                "length_buffer": lang_config.get("length_buffer", 1000),
                "guide": lang_config.get("guide", True)
            })
            
            if use_local:
                if self.local_model is None or self.local_tokenizer is None:
                    return None
                
                result = await self.benchmarker.evaluate_local_llm(
                    self.local_model, self.local_tokenizer, test_params
                )
            else:
                if self.api_client is None:
                    return None
                
                result = await self.benchmarker.evaluate_api_llm(
                    self.api_client, test_params
                )
            
            # 添加测试参数信息
            result.update({
                "test_timestamp": datetime.now().isoformat(),
                "test_params": test_params
            })
            
            return result
            
        except Exception as e:
            print(f"测试执行失败: {e}")
            return {
                "test_timestamp": datetime.now().isoformat(),
                "test_params": params,
                "error": str(e),
                "success": False,
                "score": 0.0
            }
    
    async def run_comprehensive_test(self, combinations: List[Dict], use_local: bool = True):
        """运行全面测试"""
        print(f"\n开始运行参数化测试 ({len(combinations)} 个组合)")
        print("="*80)
        
        total_tests = len(combinations)
        completed_tests = 0
        
        for i, params in enumerate(combinations, 1):
            print(f"\n测试 {i}/{total_tests}: {params['task_type']}")
            print(f"  上下文长度: {params['context_length']}")
            print(f"  埋针深度: {params['depth_percent']}%")
            print(f"  针数: {params['num_needles']}")
            print(f"  语言: {params['language']}")
            
            result = await self.run_single_test(params, use_local)
            
            if result:
                self.results.append(result)
                score = result.get('score', 0.0)
                print(f"  得分: {score:.3f}")
            else:
                print(f"  测试失败")
            
            completed_tests += 1
            progress = (completed_tests / total_tests) * 100
            print(f"  进度: {progress:.1f}%")
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.results:
            print("没有测试结果可分析")
            return
        
        print("\n" + "="*80)
        print("测试结果分析")
        print("="*80)
        
        # 按任务类型分组
        task_results = {}
        for task_type in self.parameter_space["task_types"]:
            task_results[task_type] = [r for r in self.results if r.get('test_params', {}).get('task_type') == task_type]
        
        # 总体统计
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.get('success', False)])
        avg_score = sum(r.get('score', 0.0) for r in self.results) / total_tests
        
        print(f"总体统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  成功测试数: {successful_tests}")
        print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
        print(f"  平均得分: {avg_score:.3f}")
        
        # 按任务类型统计
        print(f"\n按任务类型统计:")
        for task_type, results in task_results.items():
            if results:
                task_scores = [r.get('score', 0.0) for r in results]
                avg_task_score = sum(task_scores) / len(task_scores)
                max_score = max(task_scores)
                min_score = min(task_scores)
                print(f"  {task_type}:")
                print(f"    测试数: {len(results)}")
                print(f"    平均得分: {avg_task_score:.3f}")
                print(f"    最高得分: {max_score:.3f}")
                print(f"    最低得分: {min_score:.3f}")
        
        # 按上下文长度组统计
        print(f"\n按上下文长度组统计:")
        for group_name, group_lengths in self.parameter_space["context_length_groups"].items():
            group_results = []
            for length in group_lengths:
                group_results.extend([r for r in self.results if r.get('test_params', {}).get('context_length') == length])
            
            if group_results:
                group_scores = [r.get('score', 0.0) for r in group_results]
                avg_group_score = sum(group_scores) / len(group_scores)
                print(f"  {group_name} (lengths: {group_lengths}):")
                print(f"    测试数: {len(group_results)}")
                print(f"    平均得分: {avg_group_score:.3f}")
        
        # 按语言统计
        print(f"\n按语言统计:")
        for language in self.parameter_space["language_configs"]:
            lang_results = [r for r in self.results if r.get('test_params', {}).get('language') == language]
            if lang_results:
                lang_scores = [r.get('score', 0.0) for r in lang_results]
                avg_lang_score = sum(lang_scores) / len(lang_scores)
                print(f"  {language}:")
                print(f"    测试数: {len(lang_results)}")
                print(f"    平均得分: {avg_lang_score:.3f}")
        
        # 按埋针深度统计 - 只显示有结果的深度
        print(f"\n按埋针深度统计:")
        depth_results_map = {}
        for result in self.results:
            depth = result.get('test_params', {}).get('depth_percent')
            if depth is not None:
                if depth not in depth_results_map:
                    depth_results_map[depth] = []
                depth_results_map[depth].append(result)
        
        # 按深度排序
        for depth in sorted(depth_results_map.keys()):
            depth_results = depth_results_map[depth]
            depth_scores = [r.get('score', 0.0) for r in depth_results]
            avg_depth_score = sum(depth_scores) / len(depth_scores)
            print(f"  {depth}% 深度: {len(depth_results)} 个测试, 平均得分: {avg_depth_score:.3f}")
    
    def export_results(self, filename: str = None):
        """导出测试结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # JSON格式导出
        json_filename = f"{filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已导出到: {json_filename}")
        
        # CSV格式导出（简化版）
        csv_filename = f"{filename}.csv"
        if self.results:
            fieldnames = ['timestamp', 'task_type', 'context_length', 'depth_percent', 
                         'num_needles', 'language', 'score', 'success', 'model_response']
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'timestamp': result.get('test_timestamp', ''),
                        'task_type': result.get('task_type', ''),
                        'context_length': result.get('test_params', {}).get('context_length', ''),
                        'depth_percent': result.get('test_params', {}).get('depth_percent', ''),
                        'num_needles': result.get('test_params', {}).get('num_needles', ''),
                        'language': result.get('test_params', {}).get('language', ''),
                        'score': result.get('score', 0.0),
                        'success': result.get('success', False),
                        'model_response': result.get('response', '')[:200] if result.get('response') else ''
                    }
                    writer.writerow(row)
            
            print(f"CSV结果已导出到: {csv_filename}")

async def main():
    """主测试函数"""
    tester = ParameterizedBenchmarkTester()
    
    # 配置选项
    LOCAL_MODEL_PATH = "/path/to/your/local/model"  # 修改为实际路径
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-40d956c427e54efabdf1190630c6840e"
    
    print("参数化基准测试框架")
    print("="*80)
    print("支持多维度参数组合测试:")
    print("- 上下文长度组:")
    print("  * short: 4096, 8192")
    print("  * medium: 16384, 32768")
    print("  * long: 65536, 128000")
    print("  * extra_long: 256000, 512000")
    print("  * extreme: 1000000")
    print("- 埋针深度: 0%, 10%, 21%, 31%, 42%, 52%, 63%, 73%, 84%, 94%, 100%")
    print("- 针数配置: 根据任务类型动态调整")
    print("- 语言: 中文/英文 (带不同长度缓冲)")
    print("- 任务类型: 单针检索/多针检索/多针推理/祖源追溯")
    print("")
    
    # 选择上下文长度组
    length_group_choice = input("选择上下文长度组 (1=short, 2=medium, 3=long, 4=extra_long, 5=extreme, 6=all): ")
    length_group_map = {
        "1": "short",
        "2": "medium",
        "3": "long",
        "4": "extra_long",
        "5": "extreme",
        "6": None
    }
    length_group = length_group_map.get(length_group_choice, None)
    
    # 输入最大测试组合数
    max_combinations = input("输入最大测试组合数 (默认30): ")
    try:
        max_combinations = int(max_combinations) if max_combinations.strip() else 30
    except ValueError:
        max_combinations = 30
    
    # 生成测试组合
    combinations = tester.generate_test_combinations(max_combinations=max_combinations, length_group=length_group)
    print(f"生成了 {len(combinations)} 个测试组合")
    
    # 选择测试模式
    test_mode = input("选择测试模式 (1=本地模型, 2=DeepSeek API): ")
    
    if test_mode == "1":
        # 本地模型测试
        if LOCAL_MODEL_PATH == "/path/to/your/local/model":
            print("请先修改LOCAL_MODEL_PATH变量为您的本地模型路径")
            return
        
        if await tester.setup_local_model(LOCAL_MODEL_PATH):
            await tester.run_comprehensive_test(combinations, use_local=True)
    elif test_mode == "2":
        # API测试
        if DEEPSEEK_API_KEY == "your_api_key_here":
            print("请先设置DEEPSEEK_API_KEY环境变量或修改代码中的API密钥")
            return
        
        if tester.setup_api_client(DEEPSEEK_API_KEY):
            await tester.run_comprehensive_test(combinations, use_local=False)
    else:
        print("无效的选择")
        return
    
    # 分析结果
    tester.analyze_results()
    
    # 导出结果
    export_choice = input("是否导出测试结果? (y/n): ")
    if export_choice.lower() == 'y':
        tester.export_results()

if __name__ == "__main__":
    import torch  # 确保torch在本地模型测试时可用
    asyncio.run(main())