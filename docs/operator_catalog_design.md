# 算子目录（Operator Catalog）设计文档

## 1. 背景与目标

DeepMT 需要对 PyTorch、TensorFlow、PaddlePaddle 三个框架的常用算子进行集中管理，以支持：

- 快速检索某框架下某版本的可用算子列表；
- 为 MR 自动生成流程提供算子元数据（名称、分类、版本范围）；
- 后续批量触发 MR 生成与维护的入口（本文档描述的维护体系不包含 MR 生成逻辑，仅负责算子列表本身）。

---

## 2. 维护方式：手动 + 结构化配置

### 2.1 为什么选择手动维护

| 方案 | 优点 | 缺点 |
|------|------|------|
| **手动维护 YAML** | 精确可控、无噪声、易于版本管理 | 需人工跟进框架版本更新 |
| 全自动爬取 | 信息最新 | 易产生噪声（私有 API、废弃 API 混入）；爬取逻辑脆弱 |
| 半自动（爬取 + 人工审核）| 兼顾准确性与时效性 | 实现成本高 |

三个框架的算子集合规模有限（每个框架常用算子约 100–200 个），手动维护成本可接受，且能保证准确性。建议在每个框架发布新大版本时（通常每年 1–2 次）由维护者更新相应 YAML。

### 2.2 算子列表组织原则

- 只收录**公开 stable API**，不含 `torch._C`、`tf.experimental` 等非稳定接口；
- 优先使用**模块层路径**（如 `torch.nn.ReLU`）而非底层 C++ kernel 名称；
- 收录**使用频率高**的算子，不追求穷举（可按需扩展）。

---

## 3. 文件结构

```
mr_generator/
└── config/
    └── operator_catalog/          # 算子目录 YAML 根目录
        ├── pytorch.yaml           # PyTorch 算子目录
        ├── tensorflow.yaml        # TensorFlow/Keras 算子目录
        └── paddlepaddle.yaml      # PaddlePaddle 算子目录

mr_generator/
└── base/
    └── operator_catalog.py        # Python 查询接口
```

---

## 4. YAML 结构规范

每个 YAML 文件对应一个框架，顶层字段如下：

```yaml
framework: pytorch          # 框架标识（与文件名一致）
last_updated: "2026-03-06"  # 上次人工更新日期
description: "..."          # 简短说明

operators:                  # 算子列表
  - name: torch.nn.ReLU     # 必填：算子在文档/目录中的主标识名称
    api_type: class         # 可选：class | function（从命名约定推断）
    api_path: torch.nn.functional.relu  # 可选：用于测试的可导入完整路径
    api_style: function     # 可选：function | module | method（默认 function）
    module_class: torch.nn.ReLU         # 可选：对应的 nn.Module 类路径（元数据）
    category: activation    # 推荐：算子分类
    since: "1.0"            # 推荐：首次引入版本
    deprecated: "2.0"       # 可选：标记废弃的版本
    removed: "2.5"          # 可选：正式移除的版本
    aliases:                # 可选：其他等价 API 路径
      - torch.nn.functional.relu
    doc_url: "https://..."  # 可选：官方文档页面 URL
    signature: "(input)"    # 可选：参数签名快照（用于 check-updates 变更检测）
    note: "..."             # 可选：备注
    input_specs:            # 可选：输入参数规范（见 4.3 节）
      - name: input
        dtype: [float16, bfloat16, float32, float64]
        shape: any
        value_range: null
        required: true
    input_specs_auto: true  # 可选：true = 自动生成待确认，false/缺省 = 已人工确认
```

### 4.1 算子分类（category）

| 分类 | 说明 |
|------|------|
| `activation` | 激活函数（ReLU、GELU、Sigmoid 等）|
| `normalization` | 归一化层（BatchNorm、LayerNorm 等）|
| `pooling` | 池化层（MaxPool、AvgPool 等）|
| `convolution` | 卷积层（Conv2D、ConvTranspose 等）|
| `linear` | 线性/全连接层 |
| `recurrent` | 循环网络层（LSTM、GRU 等）|
| `transformer` | Transformer 系列（MultiheadAttention 等）|
| `dropout` | Dropout 类算子 |
| `embedding` | 嵌入层 |
| `loss` | 损失函数 |
| `math` | 数学运算（逐元素、矩阵运算等）|
| `tensor_ops` | 张量操作（拼接、切分、变形等）|
| `distance` | 距离 / 相似度度量 |
| `sparse` | 稀疏张量运算（保留，待扩展）|

### 4.3 input_specs 规范

`input_specs` 描述算子的输入张量参数，供 `InputGenerator` 生成合法测试输入。每个条目对应一个 Tensor 类型参数（跳过标量 `int`/`float`/`bool`/`str` 参数）。

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 参数名，与函数签名一致（如 `input`、`weight`） |
| `dtype` | List[str] | 支持的 dtype 列表；空列表 `[]` 表示待确认 |
| `shape` | str | 形状约束字符串（见下方） |
| `value_range` | null \| [min, max] | 值域约束；`null` 表示无约束；边界用 `null` 表示无界 |
| `required` | bool | 是否必填，默认 `true` |

#### dtype 可选值

仅使用以下 PyTorch 标准名称：

```
float16   bfloat16   float32   float64
int8      int16      int32     int64    uint8
bool      complex64  complex128
```

#### shape 语法

| 值 | 含义 |
|----|------|
| `any` | 任意形状（默认） |
| `nd>=N` | 至少 N 维（N 为正整数，如 `nd>=2`） |
| `(n,)` | 恰好 1 维 |
| `(n, m)` | 恰好 2 维 |
| `(n, c, h, w)` | 恰好 4 维（如 CNN 输入） |
| `(*, n, m)` | 最后 2 维固定，前面批次维任意 |

#### value_range 格式

```yaml
value_range: null          # 无约束
value_range: [0, null]     # 非负实数（≥ 0）
value_range: [1.0e-7, null]  # 严格正数（避免 log(0)）
value_range: [-1, 1]       # 有界区间 [-1, 1]
value_range: [null, 0]     # 非正实数（≤ 0）
```

#### input_specs_auto 标志

`input_specs_auto: true` 表示该条目的 `input_specs` 由 `deepmt catalog import-api --enrich` 自动生成，**需人工核查后修正**。确认无误后将该字段删除或设为 `false`。

`catalog info <operator>` 命令会对 `input_specs_auto: true` 的条目显示警告提示。

#### 完整示例

```yaml
- name: torch.nn.functional.relu
  api_type: function
  api_path: torch.nn.functional.relu
  api_style: function
  module_class: torch.nn.ReLU
  category: activation
  since: "1.0"
  doc_url: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
  input_specs:
    - name: input
      dtype: [float16, bfloat16, float32, float64]
      shape: any
      value_range: null
      required: true
    - name: inplace
      dtype: []          # 非 Tensor 参数，通常不在 input_specs 中
      shape: any
      value_range: null
      required: false
  input_specs_auto: false  # 已人工确认
```

注：`inplace` 等标量参数通常不需要列入 `input_specs`，示例仅供说明格式。

---

### 4.4 api_path 与 api_style

当前阶段（算子层测试）统一使用 **function 形态**：

| 字段 | 说明 |
|------|------|
| `api_path` | 可导入的完整 Python 路径（如 `torch.nn.functional.relu`），用于测试时动态调用 |
| `api_style` | `function`：直接调用；`module`：需实例化后调用 forward；`method`：绑定方法 |
| `module_class` | 对应的 `nn.Module` 子类（如 `torch.nn.ReLU`），仅作元数据，模型层使用 |

`api_path` 为空时，插件（Plugin）会尝试直接使用 `name` 字段。

---

### 4.2 版本管理语义

版本字段统一使用 `"major.minor"` 格式字符串（如 `"2.0"`、`"1.9"`）。比较规则：

```
算子在版本 V 可用  <=>  since <= V  AND  (removed 未设置 OR V < removed)
```

不强制要求 `patch` 位，忽略 patch 差异（通常算子增删在 minor 版本边界发生）。

---

## 5. Python 查询接口

`OperatorCatalog`（位于 [mr_generator/base/operator_catalog.py](../mr_generator/base/operator_catalog.py)）提供以下方法：

### 5.1 基本用法

```python
from mr_generator.base.operator_catalog import OperatorCatalog

catalog = OperatorCatalog()

# 获取 PyTorch 2.1 全部可用算子名称
names = catalog.get_operator_names("pytorch", version="2.1")

# 获取 TensorFlow 2.0 的激活函数算子
act_ops = catalog.get_by_category("tensorflow", "activation", version="2.0")

# 查询算子详情（含 since、aliases、note 等）
info = catalog.get_operator_info("pytorch", "torch.nn.ReLU")

# 判断某算子在指定版本是否可用
ok = catalog.is_available("paddlepaddle", "paddle.nn.ReLU", version="2.0")

# 各框架算子数量摘要
print(catalog.summary())
# {'pytorch': 140, 'tensorflow': 128, 'paddlepaddle': 132}
```

### 5.2 接口一览

| 方法 | 说明 |
|------|------|
| `get_all_frameworks()` | 返回所有已加载的框架名称 |
| `get_all_entries(framework, version, include_deprecated)` | 返回 `OperatorEntry` 对象列表 |
| `get_operator_names(framework, version, include_deprecated)` | 返回算子名称字符串列表 |
| `get_by_category(framework, category, version)` | 按分类过滤 |
| `get_operator_info(framework, operator_name)` | 按名称或别名查找单个条目 |
| `is_available(framework, operator_name, version)` | 判断可用性 |
| `get_categories(framework)` | 返回框架下所有出现的分类 |
| `summary(framework)` | 返回算子计数摘要 |
| `reload()` | 从磁盘热重载所有 YAML |

---

## 6. 版本差异管理策略

算子目录使用**算子为中心**的版本标注方式（而非"版本快照"方式）：

```
版本快照方式（不采用）          算子中心方式（采用）
─────────────────────          ──────────────────────
v2.0:                          paddle.nn.ReLU:
  - paddle.nn.ReLU               since: "2.0"
  - paddle.nn.Sigmoid            ...
v2.2:                          paddle.nn.Mish:
  - paddle.nn.ReLU               since: "2.2"    ← 只需在新算子处添加一行
  - paddle.nn.Sigmoid
  - paddle.nn.Mish               # 已有算子无需在每个版本重复列出
```

**优点**：新增算子只需添加一条记录；旧算子无需在每个版本重复列出；YAML 文件不随版本数量膨胀。

**版本更新时的维护操作**：

1. 新增算子 → 在对应 YAML 末尾添加条目，设置 `since` 为当前版本；
2. 废弃算子 → 在已有条目添加 `deprecated: "x.y"`；
3. 移除算子 → 在已有条目添加 `removed: "x.y"`（保留历史记录）；
4. 更新 `last_updated` 字段为当前日期。

---

## 7. 与现有代码的集成

`OperatorCatalog` 是独立模块，不修改现有代码逻辑。未来可集成到以下位置：

- **`OperatorMRGenerator`**：在批量模式下，从 `OperatorCatalog` 获取算子列表后循环调用 `generate()`；
- **`OperatorInfoFetcher`**：以目录中的算子名称作为搜索输入，提高搜索针对性；
- **`KnowledgeBase`**：结合目录中的算子名称验证 `knowledge_base.yaml` 条目是否仍对应有效算子。

---

## 8. 扩展指南

### 8.1 添加新算子

在对应 YAML 的 `operators` 列表末尾追加条目即可，无需修改 Python 代码。

### 8.2 添加新框架

1. 在 `mr_generator/config/operator_catalog/` 下新建 `<framework>.yaml`；
2. 在 `OperatorCatalog._FRAMEWORK_FILES` 中添加映射；
3. 在 `core/framework.py` 的 `SUPPORTED_FRAMEWORKS` 和 `FRAMEWORK_ALIASES` 中补充框架名称。

### 8.3 测试

```python
from mr_generator.base.operator_catalog import OperatorCatalog

catalog = OperatorCatalog()

# 验证数据加载
assert catalog.summary()["pytorch"] > 0

# 验证版本过滤
torch_2_0_names = catalog.get_operator_names("pytorch", version="2.0")
assert "torch.nn.ReLU" in torch_2_0_names
assert "torch.nn.RMSNorm" not in torch_2_0_names  # since 2.4

# 验证别名查找
entry = catalog.get_operator_info("torch", "torch.nn.functional.relu")
assert entry is not None and entry.name == "torch.nn.ReLU"
```
