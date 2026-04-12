# DeepMT 配置指南

## 一、配置文件

项目使用 `config.yaml` 进行集中配置管理。

### 查找优先级

系统按以下顺序查找配置文件（先找到即使用）：

1. 环境变量 `DEEPMT_CONFIG_PATH` 指定的路径
2. 当前工作目录的 `config.yaml`
3. 项目根目录的 `config.yaml`
4. 用户配置目录 `~/.config/deepmt/config.yaml`

### 配置项说明

```yaml
# config.yaml 完整示例

# LLM 配置（MR 自动生成时需要）
llm:
  provider: "openai"           # openai 或兼容 OpenAI API 的服务
  api_key: "sk-..."            # 也可通过 OPENAI_API_KEY 环境变量设置
  model_base: "gpt-4o-mini"   # 基础推理模型（MR 候选生成）
  model_max: "gpt-4o"         # 高级模型（形式化验证辅助，可留空）
  url: ""                      # 自定义 API 端点（留空则使用 OpenAI 默认）
  temperature: 0.2

# OCR 配置（可选，用于算子文档图片识别）
ocr:
  enabled: false
  api_key: ""
  url: ""
```

---

## 二、环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | — | OpenAI 或兼容服务的 API 密钥 |
| `DEEPMT_CONFIG_PATH` | — | 指定配置文件路径或目录 |
| `DEEPMT_LOG_LEVEL` | `INFO` | 终端日志级别（DEBUG/INFO/WARNING/ERROR） |
| `DEEPMT_LOG_CONSOLE_STYLE` | `colored` | 终端日志格式（`colored` / `file`） |
| `DEEPMT_LOG_DIR` | `data/logs` | 日志文件存储目录 |
| `DEEPMT_INJECT_FAULTS` | — | 缺陷注入规格（`all` 或 `op:mutant,...`） |

详细说明见 [`docs/environment_variables.md`](docs/environment_variables.md)。

---

## 三、快速启动

### 3.1 安装

```bash
# 完整安装（含 UI、LLM、SymPy）
pip install -e ".[all]"

# 或分开安装
pip install -e ".[llm]"       # 仅 LLM 依赖
pip install -e ".[ui]"        # 仅 Web 仪表盘依赖
pip install -e ".[sympy]"     # 仅 SymPy 依赖
```

### 3.2 配置 API 密钥

```bash
# 方式一：环境变量（推荐，不写入磁盘）
export OPENAI_API_KEY="sk-..."

# 方式二：配置文件
cp config.yaml.example config.yaml  # 若有示例文件
# 编辑 config.yaml，填入 llm.api_key
```

### 3.3 验证安装

```bash
deepmt health check
```

### 3.4 运行演示（无需 LLM）

```bash
# 运行黄金演示路径（使用预生成的 MR，约 30 秒完成）
PYTHONPATH=$(pwd) python demo/golden_path.py
```

---

## 四、数据目录结构

```
data/
├── logs/                          # 日志文件（按日期轮转，保留 14 天）
│   └── deepmt_YYYYMMDD.log
├── knowledge/
│   ├── mr_repository/operator/   # MR 知识库（每算子一个 YAML 文件）
│   ├── mr_library/                # MR 项目库（只读导出）
│   └── operator_catalog/          # 算子目录（pytorch/tensorflow/paddlepaddle）
├── results/
│   ├── evidence/                  # 缺陷证据包（JSON，含可复现脚本）
│   └── cross_framework/           # 跨框架一致性测试结果
└── cache/
    ├── web_search/                # 网络搜索结果缓存
    └── sympy/                     # SymPy 验证缓存
```

---

## 五、CLI 快速参考

```bash
# 核心工作流
deepmt mr generate torch.nn.functional.relu --save        # 单算子生成 MR
deepmt mr batch-generate --framework pytorch               # 批量生成
deepmt test batch --framework pytorch                      # 批量测试
deepmt test open --inject-faults all --collect-evidence    # 缺陷检测
deepmt test report                                         # 查看报告
deepmt test evidence list                                  # 查看证据包
deepmt ui start                                            # 启动 Web 仪表盘

# 管理
deepmt repo stats                                          # MR 知识库统计
deepmt catalog list --framework pytorch                    # 算子目录
deepmt health check                                        # 系统健康检查
```

完整 CLI 参考见 [`docs/cli_reference.md`](docs/cli_reference.md)。

---

## 六、故障排除

| 问题 | 解决方法 |
|------|---------|
| `deepmt: command not found` | `pip install -e ".[all]"` 并激活 venv |
| `LLM API key is required` | 设置 `OPENAI_API_KEY` 或在 `config.yaml` 填写 `llm.api_key` |
| MR 生成失败 | 检查网络连接；运行 `deepmt health check` 查看详情 |
| 批量测试无结果 | 运行 `deepmt repo stats` 确认知识库不为空 |
| Web 仪表盘无法启动 | 确认已安装 UI 依赖：`pip install -e ".[ui]"` |
