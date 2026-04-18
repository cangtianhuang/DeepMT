# 项目对齐改进任务单

> 生成日期：2026-04-17  
> 依据：`docs/thesis_project_alignment_analysis.md`（已清理降维表述的版本）  
> 原则：所有改进均为**项目补全**，使实现与论文声明对齐；不修改论文描述以迁就当前实现。

本文档仅记录**需要修改项目代码**的任务，纯论文文字修改不在此列。

---

## 任务总览与状态

| ID | 优先级 | 任务 | 关键文件 | 状态 |
|----|--------|------|----------|------|
| T1 | P1-H | 边界值注入（15% 极值概率） | `random_generator.py` | ✅ 完成 |
| T2 | P1-F | BenchmarkSuite 算子扩充至 53 个 | `suite.py` | ✅ 完成 |
| T3 | P1-G | 注册工业级真实模型（ResNet-18 / VGG-16 / LSTM / BERT） | `pytorch_models.py`, `model_registry.py`, `suite.py` | ✅ 完成 |
| T4 | P1-J | 自适应容差 CalculateAdaptiveTolerance | `mr_verifier.py`, `mr_prechecker.py` | ✅ 完成 |
| T5 | P0-E | MutationTester 三层扩展（模型层 + 应用层变异） | `mutation_tester.py` | ✅ 完成 |
| T6 | P1-L | ExperimentOrganizer 新接口（留存率 / 变异检出率） | `organizer.py` | ✅ 完成 |
| T7 | P1-I | 统计置信度验证 N≥100 | `organizer.py` | ✅ 完成 |
| T8 | P2-P | 算子属性标签体系（property_tags） | `mr_templates.yaml`, `mr_templates.py` | ✅ 完成 |

---

## T1：边界值注入（P1-H）

**目标**：`RandomGenerator._generate_one()` 按可配置概率（默认 15%）将生成的张量值替换为 `±inf`、`nan`、`dtype.max`、`dtype.min` 等极值，提升对溢出 / NaN 传播 / 精度降级类缺陷的检测能力。

**修改文件**：`deepmt/analysis/verification/random_generator.py`

**实施要点**：
- 新增 `boundary_injection_prob: float = 0.15` 构造参数
- `_generate_one()` 生成普通张量后，以该概率决定是否替换为极值
- 极值类型随机选取：`+inf`、`-inf`、`nan`、`dtype_max`、`dtype_min`（需按 dtype 映射）
- `generate()` 保持向后兼容签名

---

## T2：BenchmarkSuite 算子扩充至 53 个（P1-F）

**目标**：`_OPERATOR_BENCHMARK_ENTRIES` 从当前 26 个扩充至 53 个，新增「张量操作」类（tensor_op），并补充各已有类别的算子密度。

**修改文件**：`deepmt/benchmarks/suite.py`

**各类别扩充计划（总计 +27 → 53）**：

| 类别 | 当前 | 目标 | 新增算子 |
|------|------|------|---------|
| activation | 7 | 10 | elu, selu, silu |
| math | 7 | 13 | subtract, divide, floor, ceil, neg, sin |
| normalization | 2 | 3 | instance_norm |
| pooling | 2 | 4 | adaptive_avg_pool2d, adaptive_max_pool2d |
| loss | 2 | 3 | binary_cross_entropy |
| reduction | 5 | 8 | prod, var, cumsum |
| linear_algebra | 2 | 4 | conv2d, conv1d |
| tensor_op（新） | 0 | 8 | reshape, flatten, squeeze, unsqueeze, permute, cat, split, stack |

**五大类别对应关系（论文描述 → 内部代码分类）**：
- 数学运算 → `math`（13）
- 激活函数 → `activation`（10）
- 归约运算 → `reduction`（8）+ `loss`（3）= 11
- 张量操作 → `tensor_op`（8）+ `normalization`（3）= 11
- 卷积池化 → `linear_algebra`（4）+ `pooling`（4）= 8，补充 `conv2d`/`conv1d` 即可

---

## T3：注册工业级真实模型（P1-G）

**目标**：将 BenchmarkSuite 模型层对象由 SimpleMLP / SimpleCNN / SimpleRNN / TinyTransformer 替换为 ResNet-18 / VGG-16 / LSTM / BERT-base 编码器。

**修改文件**：
- `deepmt/benchmarks/models/pytorch_models.py`：新增 4 个工业模型类（懒导入 torchvision / transformers）
- `deepmt/benchmarks/models/model_registry.py`：新增 4 条 `_BENCHMARK_SPECS` 条目
- `deepmt/benchmarks/suite.py`：更新 `_MODEL_BENCHMARK_NAMES`
- `pyproject.toml`：新增可选依赖组 `benchmarks = ["torchvision>=0.10.0", "transformers>=4.0.0"]`

**注意事项**：
- Simple* 模型定义保留在 `pytorch_models.py` 中（测试用，不再出现在 BenchmarkSuite）
- 工业模型全部使用 `pretrained=False`，确保 CPU 可运行、结果可复现
- BERT 只取 encoder 部分（`BertModel.from_pretrained("bert-base-uncased", ...)`），输入为 input_ids

---

## T4：自适应容差 CalculateAdaptiveTolerance（P1-J）

**目标**：实现基于算子类型 / 输入规模自动计算容差的机制，替代当前的全局静态 `atol`。

**修改文件**：
- `deepmt/analysis/verification/mr_verifier.py`：新增 `calculate_adaptive_tolerance(operator_name, input_shapes)` 函数，`verify()` 优先使用该函数计算的动态 atol
- `deepmt/analysis/verification/mr_prechecker.py`：同步引用自适应容差

**容差规则**：

| 类型 | 规则 |
|------|------|
| 逐元素算子（relu, sigmoid 等） | `atol = 1e-6` |
| 矩阵运算（matmul, conv2d 等） | `atol = 1e-5 * sqrt(max(N, 1))`，N 为最大维度乘积 |
| 归约运算（sum, mean 等） | `atol = 1e-5 * max(dim_size, 1)` |
| fp16 输入 | 上述结果 × 10 |
| 默认 | `atol = 1e-6` |

---

## T5：MutationTester 三层扩展（P0-E）

**目标**：为 `MutationTester` 增加模型层与应用层变异类型，使三层受控变异实验对称。

**修改文件**：`deepmt/analysis/reporting/mutation_tester.py`

**新增变异类型**：

| 层次 | 变异名 | 说明 |
|------|--------|------|
| 模型层 | `WEIGHT_PERTURBATION` | forward 前对参数加高斯噪声，幅度为权重标准差的 10% |
| 模型层 | `EVAL_MODE_DISABLED` | 强制使用 training 模式运行（BN / Dropout 行为改变） |
| 应用层 | `LABEL_FLIP` | 系统性翻转 ground-truth 标签（分类任务） |
| 应用层 | `AUGMENTATION_REVERSE` | 注入与正常增强方向相反的变换（如先裁剪再 pad） |

新增 `run_model(model_ir, mutant_type, framework, n_samples)` 和 `run_application(app_ir, mutant_type, n_samples)` 入口方法，返回统一的 `MutantResult`。

---

## T6：ExperimentOrganizer 新接口（P1-L）

**目标**：在 `ExperimentOrganizer` 中新增三个指标计算接口。

**修改文件**：`deepmt/experiments/organizer.py`

**新增方法**：
```python
def compute_retention_rate(self, layer: str = "operator", method: str = "all") -> float:
    """有效留存率 = 有效 MR 数 / 初始候选数"""

def compute_mutation_score(self, layer: str = "operator", framework: str = "pytorch") -> float:
    """变异检出率，从 MutationTester 历史结果中聚合"""

def compute_stat_confidence(self, layer: str, threshold: float = 0.95, n_samples: int = 100) -> dict:
    """N≥100 统计置信度验证（T7 实现在同一方法中）"""
```

---

## T7：统计置信度验证 N≥100（P1-I）

**目标**：为已入库的 MR 提供大样本统计验证入口，通过率低于 θ=0.95 的 MR 标记为 `stat_unverified`。

**实现方式**：与 T6 合并，在 `compute_stat_confidence()` 中调用 BatchTestRunner 执行 N=100 次采样，统计通过率并更新 MR 的 `lifecycle_state`。

---

## T8：算子属性标签体系（P2-P）

**目标**：在 YAML 模板文件中为每条模板增加 `property_tags` 字段，`MRTemplatePool` 建立 tag → 模板名的反向索引，支持论文描述的 `MapPropertyTags` 步骤。

**修改文件**：
- `data/knowledge/mr_repository/mr_templates.yaml`：为每条模板添加 `property_tags: [...]`
- `deepmt/mr_generator/base/mr_templates.py`：新增 `_tag_index: Dict[str, List[str]]`，`get_templates_by_tag(tag)` 方法

**标签体系**：`commutative`、`distributive`、`monotone`、`idempotent`、`linear`、`even_symmetry`、`additive_identity`、`multiplicative_identity`

---

## 测试要求

每个任务完成后，在 `tests/unit/` 下新增或更新对应最小测试，确保功能打通（无需 LLM 或网络）。

---

## 完成记录

所有任务完成后更新本文档顶部状态表，并同步更新 `docs/dev/status.md`。
