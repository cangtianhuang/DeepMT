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

### 生成蜕变关系（MR）

```bash
# 单算子生成（需 OPENAI_API_KEY）
deepmt mr generate torch.relu --framework pytorch --save

# 批量生成（推荐）
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
# 浏览器打开 http://localhost:8000
```

## Python API

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

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API 密钥（MR 生成需要） |
| `DEEPMT_LOG_LEVEL` | 日志级别（`DEBUG`/`INFO`/`WARNING`） |
| `DEEPMT_LOG_DIR` | 日志目录（默认 `data/logs`） |

详见 `docs/environment_variables.md`。

## 完整演示路径

```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) python demo/golden_path.py
```

或参考 `docs/demo_golden_path.md` 中的 CLI 命令序列。

## 更多参考

| 文档 | 内容 |
|------|------|
| `docs/cli_reference.md` | CLI 完整命令参考 |
| `docs/environment_variables.md` | 环境变量说明 |
| `docs/demo_golden_path.md` | 答辩/演示用黄金路径 |
| `docs/dev/status.md` | 开发状态与已完成模块 |
| `docs/tech/operator_catalog.md` | 算子目录设计 |
| `docs/tech/operator_mr.md` | 算子 MR 技术细节 |
