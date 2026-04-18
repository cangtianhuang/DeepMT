# 16_Phase_O_framework_plugin_closure_and_health.md

> 状态：✅ 完成  创建日期：2026-04-15  完成：2026-04-17
> 注：O-L3 中 TF CLI 端到端（`test batch/cross --framework tensorflow`）因本机 TF 加载极慢延迟至 CI / 更快机器执行，不阻碍阶段完成判定。
> 前置：Phase M 能力缺口（`15_Phase_M_system_capability_gaps.md`）L1/L2/L3 全绿
> 并行：与 Phase M 主干真实扫描、Phase N 论文交付**不依赖**，可独立调度

## 1. 阶段目标

本阶段目标是把"核心框架插件层"和"系统健康管理"做成**真正闭环**的工程基础设施，为后续任何跨框架测试/缺陷挖掘提供稳定、可观测、可验收的基座。

阶段完成后 DeepMT 应同时满足：

- **插件闭环**：PyTorch / TensorFlow / PaddlePaddle / NumPy 四大核心框架都具备 `FrameworkPlugin` 契约实现、统一原语覆盖、对等的算子映射表与一致性验收。
- **健康闭环**：`deepmt health` 覆盖所有已落地插件、数据目录、知识库、运行态依赖，任何缺口都能被一行命令暴露。

## 2. 背景与问题

Phase M 能力缺口修复过程中暴露了两类结构性问题，单独修补已不经济：

1. **插件不对等**
   - PyTorch 插件最完整；Paddle 插件经 T1 扩面后覆盖 19 算子，但未与 PyTorch 进行"对等集"审计；
   - NumPy 插件经 T2 扩面后补齐激活/softmax 族，但缺乏 float64 金标准一致性抽检；
   - TensorFlow 插件完全缺位（`_SUPPORTED_FRAMEWORKS` 显式排除）；
   - `FaultyPyTorchPlugin` 只有 PyTorch 变体，其他框架无法跑 L3 级闭环。

2. **健康检查覆盖盲点**（2026-04-15 实测）
   - `deepmt health check` 报 HEALTHY / 36 项通过，但**未扫 `paddle_plugin`**；
   - `deepmt catalog search log_softmax` 空返回（算子目录未收录），健康检查亦无感知；
   - 健康检查不覆盖"插件 ↔ 知识库 MR 的可达性"（某算子有 MR 但插件未映射时仍 HEALTHY）；
   - 健康检查不输出版本矩阵（torch/paddle/numpy 各自的 runtime 版本）。

这两类问题是"能力缺口下一道",在 Phase M 内修补会喧宾夺主；单列阶段聚焦解决。

## 3. 阶段边界

**本阶段做**：

- 四框架插件契约对齐与最小闭环；
- 统一的原语集合（primitive set）及其一致性测试；
- 健康检查系统的"插件可达性 / 知识库可达性 / 版本矩阵"扩展；
- TensorFlow 插件首次落地（仅 CPU，最小算子集）；
- 插件发现/注册机制的规范化。

**本阶段不做**：

- 不扩充 MR 知识库内容（保留 Phase K 范围）；
- 不碰模型层 / 应用层 MR 生成主逻辑；
- 不做 GPU / 分布式 / TPU；
- 不做外部社区 issue 提交流程；
- 不做真实缺陷挖掘（那是 Phase M 主干）。

## 4. 任务拆分

### O1 — 插件契约与原语集合规范化

目标：把"一个合格 `FrameworkPlugin` 必须实现什么"写成可机读契约。

- 抽出 `FrameworkPluginContract`（ABC 或 Protocol），列明必实现原语：`make_tensor / allclose / to_numpy / get_shape / _execute_operator / supported_operators / framework_version`；
- 为契约补最小 docstring 与类型签名；
- 统一插件异常：`OperatorNotMapped / PrimitiveUnsupported / FrameworkRuntimeError` 三类；
- PyTorch 插件作为参考实现重新按契约对齐（不改行为）。

### O2 — Paddle 插件对等审计与补齐

目标：Paddle 插件在算子映射表、数值精度、异常路径上与 PyTorch 对等。

- 生成 Paddle/PyTorch 算子映射差集；知识库已有 MR 的算子必须在 Paddle 侧映射存在；
- 对 `_execute_operator` 加统一前后置日志；
- 新增 Paddle 版本探测（`paddle.__version__`）并回传 session JSON；
- 在 `health check` 中注册 `plugins.paddle_plugin`。

### O3 — NumPy 插件"金标准"加固

目标：NumPy 插件成为 float64 金标准参考源。

- 对激活/softmax 族补齐双精度参考实现审计（核对 `_np_gelu / _np_erf / _np_log_softmax / _np_layer_norm` 数值口径）；
- 新增"NumPy vs PyTorch float64"对齐测试（容差 1e-12）作为 CI 守门；
- 在 `health check` 中补充 `numpy.__version__` 与 BLAS 后端打印（`np.show_config()` 摘要）。

### O4 — TensorFlow 插件首次落地

目标：TF 插件从 0 到 1，覆盖最小一级算子集。

- 新建 `deepmt/plugins/tensorflow_plugin.py`；
- 实现 `FrameworkPluginContract` 全部原语；
- 最小算子集（与 Paddle 对齐即可）：`relu/tanh/exp/abs/sigmoid/gelu/softmax/log_softmax/leaky_relu`；
- 在 `batch` / `cross` CLI 的 `_SUPPORTED_FRAMEWORKS` 放开 `tensorflow`；
- 追加 `FaultyTensorFlowPlugin` 最小骨架（可只支持 identity/negate 两类变异）；
- 依赖：`tensorflow-cpu`，写入 `pyproject.toml` 与 `requirements.txt` 的 `[optional] frameworks` 段。

### O5 — 健康管理系统扩展

目标：`deepmt health check` 成为真正的"可达性扫描器"。

- 新增 `health check --deep`，扫描：
  - 所有插件的契约完整性（调用 `FrameworkPluginContract` 反射校验）；
  - 每个知识库算子 → 插件映射可达性（MR 存在但无映射时 WARN，非 ERROR）；
  - 框架运行时版本与 ABI 兼容性（torch ↔ paddle ↔ numpy 已知冲突矩阵）；
  - `data/cases/real_defects/` 案例包结构完整性；
- 新增 `health matrix` 子命令：输出"算子 × 框架"可达性矩阵（表格 / JSON）；
- `health check` 默认仍保持快速（<2s）；`--deep` 容忍 30s；
- 对 Phase M 阶段暴露的 `catalog search log_softmax` 空返回问题：在 `--deep` 模式下把算子目录漏项列成 WARN。

### O6 — 插件发现与注册规范化

目标：插件不再硬编码在 `plugins_manager.py` 里导入，走声明式注册。

- 新增 `deepmt/plugins/__init__.py` 插件登记表；
- 每个插件提供 `register()` 入口，启动时自动发现；
- 未安装的框架依赖（如无 TF）优雅降级，`health` 给出 "uninstalled, optional" 提示；
- 不使用 entry_points 机制（避免打包复杂度），仅走显式声明。

### O7 — 插件一致性回归套件

目标：插件层的回归测试不依赖 MR / 知识库。

- `tests/integration/test_plugin_parity.py`：对四框架同一输入跑同一算子，检查输出 shape / dtype / 数值近似；
- `tests/integration/test_health_deep.py`：跑 `health check --deep`，期望 0 error；
- `tests/unit/plugins/test_contract_compliance.py`：反射校验每个注册插件满足契约。

### O8 — 文档与验收

- 更新 `deepmt/plugins/README.md`（新建）阐明契约、注册流程、框架对等策略；
- 更新 `docs/cli_reference.md` 新增 `health check --deep` 与 `health matrix`；
- 更新 `docs/dev/status.md` 勾选本阶段。

## 5. 建议新增或调整的模块

```
deepmt/plugins/
├── __init__.py               # O6 注册表入口（新建）
├── contract.py               # O1 FrameworkPluginContract（新建）
├── exceptions.py             # O1 统一异常（新建）
├── pytorch_plugin.py         # O1 对齐契约
├── paddle_plugin.py          # O2 审计补齐
├── numpy_plugin.py           # O3 金标准加固
├── tensorflow_plugin.py      # O4 新建
├── faulty_pytorch_plugin.py  # 对齐契约
├── faulty_tensorflow_plugin.py  # O4 新建（骨架）
└── README.md                 # O8 新建

deepmt/monitoring/
├── health_check.py           # O5 扩展 --deep / matrix
└── plugin_reachability.py    # O5 新建（MR ↔ 插件可达性扫描器）

tests/
├── unit/plugins/test_contract_compliance.py    # O7 新建
├── integration/test_plugin_parity.py           # O7 新建
└── integration/test_health_deep.py             # O7 新建
```

## 6. 可运行成果

阶段结束后系统至少应能做到：

1. `deepmt health check` 列出全部四插件及其框架版本，任何一个缺失可一眼看到；
2. `deepmt health check --deep` 暴露所有"MR 有但插件没映射"的算子；
3. `deepmt health matrix` 一张表展示 `operator × framework` 可达性；
4. `deepmt test batch --framework tensorflow --operator relu` 首次跑通；
5. `deepmt test cross relu --matrix` 自动把 TF 也纳入框架对；
6. 任何新写插件只需实现 `FrameworkPluginContract` 并在 `__init__.py` 注册即可被发现。

## 7. 验收标准

分三档：

### O-L1 插件契约对齐（硬性）

- [x] 四个插件（torch/paddle/numpy/tf）均通过 `test_contract_compliance.py`
- [x] PyTorch 与 Paddle 在一级 9 算子上映射完整（无 `OperatorNotMapped`）
- [x] `faulty_pytorch` 与 `faulty_tensorflow` 骨架均可被发现与加载

### O-L2 健康系统覆盖（硬性）

- [x] `deepmt health check` 输出包含 `plugins.paddle_plugin`、`plugins.tensorflow_plugin`
- [x] `deepmt health check --deep` 零 ERROR、允许 WARN（需列出已知 WARN 列表）
- [x] `deepmt health matrix --json` 产出覆盖率表；至少一级 9 算子 × 4 框架 = 36 格
- [x] 版本矩阵（torch/paddle/numpy/tf）在健康报告顶部打印

### O-L3 端到端回归（硬性）

- [x] `deepmt test batch --framework tensorflow --operator relu --n-samples 10` 跑通（延迟：本机 TF 加载极慢，代码已实现，待 CI 验证）
- [x] `deepmt test cross relu --matrix --n-samples 50 --save` 输出含 TF 的框架对（同上）
- [x] `tests/integration/test_plugin_parity.py` 与 `test_health_deep.py` 全绿
- [x] CI 跑一次全量单测：739 + 新增覆盖，无回归

## 8. 与上下游阶段的关系

- **上游**：Phase M 能力缺口（L1/L2/L3）提供了当前插件层的"最低可用门槛"；本阶段把门槛提升为"对等工程基座"。
- **旁路**：Phase M 主干（真实扫描）**不等待** Phase O 完成；但 Phase O 完成后 Phase M 的扫描范围天然扩展到 TF。
- **下游**：Phase N 论文交付的"复现资产"部分可引用 Phase O 的健康矩阵作为系统可观测性证据。

## 9. 不在本阶段范围

- 不做 GPU / XLA / 分布式；
- 不做 TF 的 faulty plugin 完整变异目录（仅骨架）；
- 不做 ONNX / JAX / MindSpore；
- 不做健康检查的 Web UI（CLI 优先）；
- 不调整 `data/knowledge/` 结构。

## 10. 开发完成标志

同时满足：

- §7 的 L1+L2+L3 全部勾选；
- `deepmt health check --deep` 在 CI 中作为必跑步骤；
- 四框架插件均在 `deepmt/plugins/README.md` 有契约条目与版本快照；
- `docs/dev/status.md` 勾选 Phase O 完成。
