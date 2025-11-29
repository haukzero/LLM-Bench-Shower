# 后端测试说明

快速测试脚本用于使用 mock LLM 来快速验证 benchmark 类的功能，无需加载真实模型或调用 API。

## 概述

该测试套件包含以下内容：

### 📁 文件结构

```
tests/
├── __init__.py                      # 包初始化文件
├── test_benchmarks.py               # 完整的测试脚本（支持详细选项）
├── quick_test.py                    # 快速测试脚本（简化版）
├── fixtures/
│   ├── __init__.py
│   ├── mock_llm.py                  # Mock LLM 实现
│   └── dataset_setup.py             # 测试数据集设置
└── test_data/                       # 测试数据目录（运行时生成）
    ├── LongBench/
    ├── LongBenchV2/
    └── C-Eval/
```

## 组件说明

### 1. Mock LLM (`fixtures/mock_llm.py`)

提供以下 mock 类：

- **MockTokenizer**: 模拟 `transformers.AutoTokenizer`
  - 支持 `__call__()` 进行 tokenization
  - 支持 `decode()` 进行反向转换
  - 返回 mock tensor 对象

- **MockModel**: 模拟 `transformers.AutoModelForCausalLM`
  - 支持 `generate()` 方法生成 token
  - 支持 `to(device)` 移动到不同设备
  - 快速返回结果，无需实际计算

- **MockOpenAIClient**: 模拟 OpenAI API 客户端
  - 支持 `messages.create()` API 调用
  - 返回 mock API 响应
  - 确定性生成响应内容

### 2. 数据集设置 (`fixtures/dataset_setup.py`)

- **`setup_test_datasets()`**: 创建 mock 数据集
  - 为 LongBench, LongBenchV2, C-Eval 创建 JSONL 格式的测试数据
  - 支持自定义样本数量
  - 自动生成合理的测试数据

- **`update_config_for_testing()`**: 更新配置指向测试数据
  - 修改 `dataset_paths.json` 指向测试数据目录
  - 保存原始配置为备份

- **`cleanup_test_data()`**: 清理测试数据

### 3. 测试脚本

#### 完整版 (`test_benchmarks.py`)

功能完整的测试脚本，支持详细的命令行选项。

**基本用法：**

```bash
# 测试所有 benchmark
python tests/test_benchmarks.py

# 测试特定 benchmark
python tests/test_benchmarks.py --bench LongBench

# 测试本地模型评估
python tests/test_benchmarks.py --mode local

# 测试 API 模型评估
python tests/test_benchmarks.py --mode api

# 测试本地和 API 两种模式
python tests/test_benchmarks.py --mode all

# 自定义样本数量
python tests/test_benchmarks.py --samples 10

# 保存测试结果到指定文件
python tests/test_benchmarks.py --output my_results.json

# 测试后自动清理测试数据
python tests/test_benchmarks.py --cleanup
```

**完整选项：**

```
usage: test_benchmarks.py [-h] [--bench BENCH] [--samples SAMPLES] 
                          [--mode {local,api,all}] [--output OUTPUT] 
                          [--cleanup] [--verbose]

optional arguments:
  -h, --help              Show help message
  --bench BENCH           Specific benchmarker to test (e.g., 'LongBench')
  --samples SAMPLES       Number of samples per subdataset (default: 5)
  --mode {local,api,all}  Test mode: local/api/all (default: local)
  --output OUTPUT         Output file for results (default: test_results.json)
  --cleanup               Clean up test data after tests
  --verbose               Print detailed output
```

#### 快速版 (`quick_test.py`)

简化版脚本，用于快速测试单个或多个 benchmark。

**基本用法：**

```bash
# 测试所有 benchmark
python tests/quick_test.py

# 快速测试 LongBench
python tests/quick_test.py longbench

# 快速测试 LongBenchV2
python tests/quick_test.py longbench_v2

# 快速测试 C-Eval
python tests/quick_test.py c-eval

# 自定义样本数量
python tests/quick_test.py longbench --samples 10
```

## 测试流程

### 自动流程

脚本会自动执行以下步骤：

1. **环境设置**
   - 创建测试数据目录
   - 生成 mock 数据集（JSONL 格式）
   - 更新配置文件指向测试数据

2. **测试执行**
   - 为每个 benchmark 创建 mock 模型和客户端
   - 调用 `evaluate_local_llm()` 测试本地模型评估
   - 调用 `evaluate_api_llm()` 测试 API 模型评估
   - 收集结果

3. **结果输出**
   - 打印详细的测试摘要
   - 保存 JSON 格式的测试结果
   - 可选：清理测试数据

### 手动步骤

如果你想手动运行测试，可以参考以下步骤：

```python
from fixtures.mock_llm import create_mock_model, create_mock_tokenizer, create_mock_client
from fixtures.dataset_setup import setup_test_datasets, update_config_for_testing
from bench import init_all_benchmarkers

# 1. 设置
setup_test_datasets(num_samples=5)
update_config_for_testing()

# 2. 获取 benchmarker
benchmarkers = init_all_benchmarkers()
benchmarker = benchmarkers["LongBench"]

# 3. 测试本地模型
model = create_mock_model()
tokenizer = create_mock_tokenizer()
result = benchmarker.evaluate_local_llm(
    model=model,
    tokenizer=tokenizer,
    subdataset_name="2wikimqa"
)

# 4. 测试 API 模型
client = create_mock_client()
result = benchmarker.evaluate_api_llm(
    client=client,
    model="gpt-4",
    subdataset_name="2wikimqa"
)
```

## 支持的 Benchmark

目前支持以下 benchmark 的快速测试：

### LongBench
- 长文本理解评估
- 子数据集：2wikimqa, dureader, gov_report, hotpotqa, narrativeqa, ...
- 测试数据包含：question, context, answers

### LongBenchV2
- 改进的长文本理解评估
- 领域：Code Repository Understanding, Long Dialogue History, In-context Learning, ...
- 测试数据包含：question, instruction, context, answer

### C-Eval
- 中文多学科评估（placeholder）
- 测试数据包含：question, options (A/B/C/D), answer

## 测试结果

### 结果格式

脚本默认将结果保存为 `test_results.json`，格式如下：

```json
{
  "LongBench": {
    "benchmarker": "LongBench",
    "local_tests": {
      "2wikimqa": {
        "dataset": "2wikimqa",
        "model_type": "local",
        "metrics": {
          "total": 5,
          "processed": 5
        },
        "predictions": [...]
      }
    },
    "api_tests": {
      "2wikimqa": {
        "dataset": "2wikimqa",
        "model_type": "api",
        "model": "mock-gpt-4",
        "metrics": {
          "total": 5,
          "processed": 5
        },
        "predictions": [...]
      }
    }
  }
}
```

### 输出示例

```
======================================================================
🚀 Setting up test environment
======================================================================

📁 Setting up mock datasets in: /path/to/tests/test_data

✓ Created mock LongBench dataset with 5 subdatasets
✓ Created mock LongBenchV2 dataset with 3 domains
✓ Created mock C-Eval dataset with 3 subjects

✓ All mock datasets created successfully

✓ Updated config: /path/to/LLMBenchShower/configs/dataset_paths.json

======================================================================
Testing Benchmarker: LongBench
======================================================================

Available subdatasets: 6
Testing subdataset: 2wikimqa

📝 Testing local LLM evaluation: LongBench/2wikimqa
✓ Local LLM evaluation completed
  - Total samples: 5
  - Processed: 5

📡 Testing API LLM evaluation: LongBench/2wikimqa
✓ API LLM evaluation completed
  - Total samples: 5
  - Processed: 5

======================================================================
📊 TEST SUMMARY
======================================================================

✓ LongBench
  Local LLM tests:
    ✓ 2wikimqa: 5/5 samples processed
  API LLM tests:
    ✓ 2wikimqa: 5/5 samples processed

...

✓ Testing complete!
```

## 常见问题

### Q: 为什么需要 mock LLM？
A: Mock LLM 允许快速测试代码逻辑而无需：
- 下载和加载真实的大型语言模型（通常需要几 GB 的内存）
- 调用实际的 API（需要有效的 API key，可能产生费用）
- 等待真实的推理时间（可能需要数秒到数分钟）

这样可以在开发和 CI/CD 流程中快速迭代。

### Q: Mock LLM 返回的是真实的答案吗？
A: 不是。Mock LLM 返回确定性的模拟响应，用于测试代码流程和结果格式。
要获得真实的评估结果，需要使用实际的模型。

### Q: 如何测试我新添加的 benchmark？
A: 
1. 在 `fixtures/dataset_setup.py` 中添加你的 benchmark 的数据生成函数
2. 在 `setup_test_datasets()` 中调用该函数
3. 运行测试脚本：`python tests/test_benchmarks.py --bench YourBench`

### Q: 如何自定义 mock 数据？
A: 修改 `fixtures/dataset_setup.py` 中相应的数据生成函数。
例如，要改变样本数量，使用 `--samples` 参数。

### Q: 测试会修改原始配置吗？
A: 脚本会修改 `dataset_paths.json`，但会创建 `.backup` 备份文件。
你可以手动恢复或删除备份文件后重新运行脚本。

## 最佳实践

1. **开发阶段**：使用 `quick_test.py` 快速验证新代码
2. **CI/CD 流程**：使用 `test_benchmarks.py` 进行完整测试
3. **调试**：使用 `--verbose` 选项查看详细输出
4. **清理**：使用 `--cleanup` 选项在测试后自动清理

## 扩展

如果需要为其他 benchmark 添加支持，参考以下步骤：

1. **添加 mock 数据生成**：在 `fixtures/dataset_setup.py` 中添加函数
   ```python
   def create_mock_mybench_dataset(test_data_dir: str, num_samples: int = 5):
       # 创建 JSONL 格式的测试数据
       pass
   ```

2. **在 `setup_test_datasets()` 中调用**：
   ```python
   create_mock_mybench_dataset(test_data_dir, num_samples)
   ```

3. **更新配置映射**（如需要）：在 `update_config_for_testing()` 中添加路径

## 许可证

与主项目相同。
