# DeepMT 快速上手

## 安装

```bash
pip install -e ".[all]"
# 或仅安装基础依赖
pip install -e "."
```

## CLI 快速开始

### 健康检查

```bash
deepmt health check
```

### 查看算子目录

```bash
deepmt catalog list --framework pytorch
deepmt catalog info torch.relu
```

### 生成蜕变关系（算子层）

```bash
# 单算子生成（仅模板，无需 LLM）
deepmt mr generate torch.relu --sources template --save

# 单算子生成（LLM + 模板，需 OPENAI_API_KEY）
deepmt mr generate torch.relu --sources llm,template --precheck --save

# 批量生成
deepmt mr batch-generate --framework pytorch --limit 10
```

### 运行测试

```bash
# 批量蜕变测试（从知识库加载 MR，RandomGenerator 自动生成输入）
deepmt test batch --operator torch.relu --framework pytorch --n-samples 50

# 查看测试历史
deepmt test history

# 查看失败用例
deepmt test failures
```

### 生成报告

```bash
deepmt test report
deepmt test evidence list
```

### 启动 Web 仪表盘

```bash
deepmt ui start
# 浏览器打开 http://localhost:8080
```

---

## Python API

### 算子层测试

```python
from deepmt import DeepMT

client = DeepMT()

# 批量测试某算子（MR 已预先生成并存入知识库）
result = client.run_batch_test("torch.relu", framework="pytorch", n_samples=10)
print(result.summary())

# 查询历史
history = client.get_test_history()
failures = client.get_failed_tests(limit=10)
```

### 模型层 MR 生成

```python
from deepmt.benchmarks.models import ModelBenchmarkRegistry
from deepmt.mr_generator.model import ModelMRGenerator
from deepmt.analysis.model_verifier import ModelVerifier
import torch

registry = ModelBenchmarkRegistry()
ir = registry.get("ResNet18", with_instance=True)   # 包含 model_instance（懒加载 torchvision）

gen = ModelMRGenerator()
mrs = gen.generate(ir)   # 基于结构分析选择变换策略

# 手动验证
verifier = ModelVerifier()
x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    orig = ir.model_instance(x)
for mr in mrs:
    trans_x = eval(mr.transform_code)(x)
    with torch.no_grad():
        trans = ir.model_instance(trans_x)
    result = verifier.verify(orig, trans, mr)
    print(mr.description, result.passed)
```

工业级基准模型：`ResNet18`、`VGG16`、`LSTMBenchmark`、`BERTEncoder`（需 `pip install -e ".[benchmarks]"`）

### 应用层 MR 生成与验证

```python
from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
from deepmt.mr_generator.application import ApplicationMRGenerator
from deepmt.analysis.semantic_mr_validator import SemanticMRValidator
from deepmt.analysis.application_reporter import ApplicationReporter

registry = ApplicationBenchmarkRegistry()
sc = registry.get("TextSentiment")   # 或 "ImageClassification"

# 模板模式（不依赖 LLM）
gen = ApplicationMRGenerator(use_llm=False)
mrs = gen.generate_from_scenario("TextSentiment")

# 语义验证（mock 预测函数，或传入真实 predict_fn）
validator = SemanticMRValidator()
results = validator.validate_batch(mrs, sc.sample_inputs, sc.sample_labels)

# 人工复核
validator.review_mr(mrs[0], results[0], approved=True, note="经确认有效")

# 生成报告
reporter = ApplicationReporter()
report = reporter.generate(mrs, results, scenario_name="TextSentiment")
print(reporter.format_text(report))

# 导出为 dict（供仪表盘或论文展示）
data = reporter.to_dict(report)
```

可用应用场景：`ImageClassification`（图像分类）、`TextSentiment`（文本情感分析）

---

## 环境变量

| 变量               | 说明                                 |
| ------------------ | ------------------------------------ |
| `OPENAI_API_KEY`   | LLM API 密钥（MR 生成需要）          |
| `DEEPMT_LOG_LEVEL` | 日志级别（`DEBUG`/`INFO`/`WARNING`） |
| `DEEPMT_LOG_DIR`   | 日志目录（默认 `data/logs`）         |

详见 `docs/environment_variables.md`。

---

## 完整演示路径

```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) python demo/golden_path.py
```

或参考 `docs/demo_golden_path.md` 中的 CLI 命令序列。

---

## 更多参考

| 文档                            | 内容                 |
| ------------------------------- | -------------------- |
| `docs/cli_reference.md`         | CLI 完整命令参考     |
| `docs/environment_variables.md` | 环境变量说明         |
| `docs/demo_golden_path.md`      | 答辩/演示用黄金路径  |
| `docs/dev/status.md`            | 开发状态与已完成模块 |
| `docs/tech/operator_catalog.md` | 算子目录设计         |
| `docs/tech/operator_mr.md`      | 算子 MR 技术细节     |
