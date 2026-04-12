# DeepMT 黄金演示路径（Golden Demo Path）

> **适用场景**：论文答辩、导师汇报、项目演示  
> **前提**：已完成安装（`pip install -e ".[all]"`），有 PyTorch 环境，无需 LLM API  
> **预计时长**：约 2 分钟（全量 CLI 序列）

---

## 一、一键 Python 演示（最快，30 秒）

```bash
source .venv/bin/activate
PYTHONPATH=$(pwd) python demo/golden_path.py
```

该脚本自动运行下方 CLI 序列对应的全部 5 个步骤，含完整输出与说明。

---

## 二、CLI 命令序列（推荐答辩使用）

以下 5 步命令展示了 DeepMT 的完整核心价值链。**演示时按顺序运行，无需修改**。

### 前置：激活虚拟环境

```bash
source .venv/bin/activate
```

---

### Step 1 — 算子目录与 MR 知识库

```bash
# 查看 MR 知识库概况（已有算子数、MR 总数）
deepmt repo stats

# 列出 relu 的已验证 MR
deepmt repo list --operator torch.nn.functional.relu
```

**关键展示点**：系统已自动从算子语义中归纳出蜕变关系并存入知识库。

---

### Step 2 — 正常批量测试（基线验证）

```bash
deepmt test batch --framework pytorch --n-samples 10
```

**预期输出**：3 个演示算子（relu / exp / abs）全部 `[PASS]`。

**关键展示点**：正常 PyTorch 实现满足所有已验证蜕变关系（通过率 100%）。

---

### Step 3 — 开放测试（缺陷注入 + 检测）

```bash
deepmt test open \
  --inject-faults all \
  --n-samples 10 \
  --collect-evidence
```

**预期输出**：`torch.nn.functional.relu` 被标记 `[FAIL]`，两条 MR 均检出缺陷，生成证据包 ID。

**关键展示点**：
- 缺陷为「relu 负值截断阈值偏移」（`clamp(min=-1e-3)` 而非 `clamp(min=0)`），普通数值测试极难发现
- DeepMT 的蜕变关系在 10 次采样内 100% 检出

---

### Step 4 — 测试报告

```bash
# 总览报告（算子 / MR 级通过率）
deepmt test report

# 查看失败详情
deepmt test failures --limit 5
```

**关键展示点**：系统自动汇总每条 MR 的通过率，失败案例含数值偏差量化信息。

---

### Step 5 — 可复现证据包

```bash
# 列出所有证据包
deepmt test evidence list

# 查看第一个证据包详情（替换 <ID> 为上条命令输出的 ID）
deepmt test evidence show <ID>

# 输出可直接运行的 Python 复现脚本
deepmt test evidence script <ID>
```

**关键展示点**：每个缺陷案例都能输出一段完整的 Python 脚本，粘贴即可复现缺陷，这是研究可信度的核心体现。

---

### （可选）Step 6 — Web 仪表盘

```bash
deepmt ui start
# 浏览器打开 http://localhost:8000
```

---

## 三、演示数据说明

| 算子 | 预置 MR 数 | 注入缺陷类型 | MR 能否检出 |
|------|-----------|------------|-----------|
| `torch.nn.functional.relu` | 2 条（已验证） | `add_const`（截断阈值偏移） | ✅ 全部检出 |
| `torch.exp` | 2 条（1 已验证） | `scale`（1.001 倍近似误差） | ⚠ 当前 MR 容差内通过，需补充 MR |
| `torch.abs` | 1 条（已验证） | `scale`（0.999 倍量化误差） | ⚠ 当前 MR 容差内通过，需补充 MR |

> `torch.exp` 和 `torch.abs` 的"漏检"是**正常且有意义的研究结论**：说明现有 MR 对微小数值缺陷的灵敏度有限，需要补充更严格的 MR（如绝对精度检验），体现了 MR 质量对缺陷检测能力的影响（RQ2）。

---

## 四、故障排除

| 问题 | 排查方向 |
|------|---------|
| `No MRs found` | 检查 `data/knowledge/mr_repository/operator/` 目录是否存在对应 YAML 文件 |
| `Plugin not found` | 确认已安装 PyTorch：`pip install torch` |
| `deepmt: command not found` | 确认已安装项目：`pip install -e ".[all]"` 并激活 venv |
| 批量测试 0 条 | 运行 `deepmt repo stats` 确认知识库不为空 |

---

## 五、扩展演示（需 LLM API）

如需演示 MR **自动生成**过程（需要 `OPENAI_API_KEY` 或兼容接口）：

```bash
# 单算子生成 MR 并保存到知识库
deepmt mr generate torch.nn.functional.relu --framework pytorch --save

# 批量生成（多算子，断点续跑）
deepmt mr batch-generate --framework pytorch --category activation
```

生成后重新运行 Step 2~5 即可看到更多算子和 MR 参与测试。
