# DeepMT 项目演示视频脚本

> **目标读者**：导师汇报 / 项目演示视频录制  
> **前提**：已激活虚拟环境，无需 LLM API，所有步骤均使用 `deepmt` 命令  
> **预计录制时长**：5~8 分钟

---

## 准备工作：激活环境

```bash
source .venv/bin/activate
```

---

## 第 0 步：清理已有数据（演示前必做）

清理旧日志、测试证据和所有已生成的 MR，确保从干净状态开始演示。

### 0-1 清理日志

```bash
deepmt data clean-logs -y
```

### 0-2 删除知识库中已有的所有 MR

```bash
deepmt repo delete abs          --all -y
deepmt repo delete exp          --all -y
deepmt repo delete gelu         --all -y
deepmt repo delete leaky_relu   --all -y
deepmt repo delete log_softmax  --all -y
deepmt repo delete relu         --all -y
deepmt repo delete sigmoid      --all -y
deepmt repo delete softmax      --all -y
deepmt repo delete tanh         --all -y
deepmt repo delete torch.asinh  --all -y
deepmt repo delete torch.cosh   --all -y
deepmt repo delete torch.expm1  --all -y
```

### 0-3 确认已清空

```bash
deepmt repo stats
```

> 预期输出：`总数: 0`，确认知识库为空。

---

## 第 1 步：系统健康检查

```bash
deepmt health check
```

**演示重点**：展示框架版本矩阵（PyTorch / NumPy 等），确认系统就绪。

---

## 第 2 步：浏览算子目录

```bash
# 查看 PyTorch 激活函数类算子
deepmt catalog list --framework pytorch --category activation

# 查询 relu 的跨框架分布
deepmt catalog info relu
```

**演示重点**：系统内置算子知识库，涵盖 PyTorch / TensorFlow / PaddlePaddle 三大框架。

---

## 第 3 步：为算子自动生成蜕变关系（MR）

```bash
# 为 relu 生成 MR（模板来源，带数值预检，保存到知识库）
deepmt mr generate relu --sources template --precheck --save

# 为 exp 生成 MR
deepmt mr generate exp --sources template --precheck --save

# 为 abs 生成 MR
deepmt mr generate abs --sources template --precheck --save
```

**演示重点**：系统从算子语义中自动归纳蜕变关系（如"relu 的非负性"、"exp 的单调性"），无需人工编写，生成后通过数值预检验证正确性。

---

## 第 4 步：查看知识库

```bash
# 查看全库概况
deepmt repo stats

# 查看 relu 的 MR 详情
deepmt repo info relu
```

**演示重点**：生成的 MR 结构化存入知识库，包含变换代码（`transform_code`）和断言表达式（`oracle_expr`）。

---

## 第 5 步：批量蜕变测试（正常验证）

```bash
deepmt test batch --framework pytorch --n-samples 10
```

**演示重点**：正常的 PyTorch 实现满足所有蜕变关系，通过率 100%。

---

## 第 6 步：缺陷注入 + 自动检测（核心亮点）

```bash
deepmt test open \
  --inject-faults all \
  --n-samples 10 \
  --collect-evidence
```

**演示重点**（这是最关键的一步）：
- 系统向 relu 注入一个「截断阈值偏移」缺陷（`clamp(min=-1e-3)` 而非 `clamp(min=0)`），这种微小偏差在普通数值测试中极难发现
- DeepMT 在 10 次采样内 100% 检出该缺陷，并自动保存可复现的证据包

---

## 第 7 步：查看测试报告

```bash
# 总览报告（按算子 / MR 显示通过率）
deepmt test report

# 查看失败详情
deepmt test failures --limit 5
```

**演示重点**：失败案例含精确的数值偏差量化信息，可定位缺陷位置。

---

## 第 8 步：可复现证据包

```bash
# 列出所有证据包
deepmt test evidence list

# 查看证据包详情（将 <ID> 替换为上一步输出的 ID）
deepmt test evidence show <ID>

# 导出可直接运行的 Python 复现脚本
deepmt test evidence script <ID>
```

**演示重点**：每个缺陷案例都有一段独立的 Python 脚本，粘贴即可复现，体现研究的可信度与可验证性。

---

## （可选）第 9 步：Web 可视化仪表盘

```bash
deepmt ui start
```

浏览器打开 `http://127.0.0.1:8080`，可视化展示实验数据、MR 知识库和测试结果。

---

## 演示结构总览

| 步骤 | 命令 | 核心展示点 |
|------|------|-----------|
| 0. 清理数据 | `repo delete` / `data clean-logs` | 从干净状态开始 |
| 1. 健康检查 | `health check` | 系统就绪、框架版本 |
| 2. 算子目录 | `catalog list` / `catalog info` | 内置知识库 |
| 3. MR 生成 | `mr generate` | 自动归纳蜕变关系（核心贡献） |
| 4. 知识库查看 | `repo stats` / `repo info` | MR 结构化存储 |
| 5. 正常测试 | `test batch` | 基线验证通过 |
| 6. 缺陷检测 | `test open` | **自动检出微小缺陷**（研究亮点） |
| 7. 测试报告 | `test report` / `test failures` | 量化分析 |
| 8. 证据包 | `test evidence` | 可复现性 |
| 9. 仪表盘 | `ui start` | 可视化展示（可选） |
