# 15_Phase_M_system_capability_gaps.md

> 状态：⬜ 计划  日期：2026-04-14  
> 前置：Phase M（`13_Phase_M_真实缺陷挖掘与案例沉淀.md`）

## 1. 背景

Phase M 再实验（用纯 `deepmt` CLI 扩面，9 算子 × 2 参考框架 × n=500 = 约 7000 次跨框架执行）结果：

- 能跑通的配置里，系统只自主发现 1 个候选差异（`tanh` 单调性在 paddle 上 5/500 违反）；
- 大量有价值的路径被 CLI / 插件层阻塞；
- 手写脚本（非 deepmt 流水线）发现的 Paddle `log_softmax` 截断缺陷，**在当前 CLI 下根本无法被发现**，原因全部是本阶段列出的系统缺口。

结论：**继续扩面测试的瓶颈不在"测试样本够不够多"，而在"CLI 能触达的表面够不够大"**。本阶段在 Phase N 之前，先把挡在 Phase M 前的系统能力补齐。

## 2. 能力缺口清单（按优先级）

### G1  Paddle 插件算子映射表覆盖面窄【阻塞-高】

**现象**：

```
deepmt test cross log_softmax --framework2 paddle
# -> PaddlePlugin: 算子 'log_softmax' 无对应实现
```

**根因**：`deepmt/plugins/paddle_plugin.py::_build_paddle_operators` 只注册了 relu/sigmoid/tanh/gelu/exp/abs/leaky_relu/softmax 等基础激活。知识库里已有的 `log_softmax` 就已经打不通。

**修复**：扩展映射表至少覆盖知识库中全部 9 个算子及 Phase N 候选高风险算子：
`log_softmax, layer_norm, batch_norm(eval), logsumexp, cos, sin, sqrt, log, reciprocal, cumsum, var, std`。
配套：`deepmt catalog search <op>` 成功返回 paddle 等价时，`test cross` 应保证 resolvable。

### G2  NumPy 插件算子映射表覆盖面窄【阻塞-高】

**现象**：

```
deepmt test cross gelu     --framework2 numpy   # unsupported_f2=500/500
deepmt test cross leaky_relu --framework2 numpy # exception_f2=500/500
```

**根因**：`deepmt/plugins/numpy_plugin.py::_NUMPY_OPERATORS` 只注册了若干函数式算子；`gelu/leaky_relu/softmax/log_softmax` 等需要数学闭式参考的不在表内。

**修复**：对激活与 softmax 族补齐 numpy 参考实现（纯 `np.*` 不依赖 scipy；需要时用 `erf` 等价公式）。此表同时天然成为"float64 金标准"，修复 `02_candidate_cases.md` 里"缺乏独立真值源"问题。

### G3  `deepmt test batch` 硬拒 paddlepaddle 主框架【阻塞-高】

**现象**：

```
deepmt test batch --framework paddlepaddle --collect-evidence
# -> 功能未实现：--framework paddlepaddle
```

**根因**：`deepmt/commands/test/execution/batch.py` 里保留着 `_SUPPORTED_FRAMEWORKS = {"pytorch"}` 门卫；而 `PaddlePlugin` 实际已存在。

**修复**：在 paddle 插件具备必要原语（`make_tensor / _execute_operator / allclose / to_numpy / get_shape`）的前提下，放开 batch 入口对 paddlepaddle 的支持。TensorFlow 保持 `NotImplementedError`。

### G4  `deepmt test cross` 丢失失败样本原始输入【阻塞-中】

**现象**：cross session JSON 只保留 `diff_type_counts: {behavior_diff: 5}` 聚合，**没有失败样本对应的输入张量**；要最小复现必须手写脚本。

**根因**：`CrossFrameworkTester` 在 `inconsistent_cases` 计数时丢掉了 tensor payload，只回灌聚合统计。

**修复**：cross session JSON 增加 `failed_samples: [{mr_id, seed, input_summary, f1_out, f2_out, diff_type}, ...]`（可设 `--keep-samples N` 限流，默认 50）。这样跨框架差异可以直接进入 `deepmt case build --from-evidence` 归档流水线。

### G5  证据包与案例构建的跨框架扩展【阻塞-中】

**现象**：`EvidencePack` 结构当前假设"单框架 oracle 违反"；跨框架差异（两边 oracle 都通过但数值不一致 / 一边通过一边不通过）无处写入。

**根因**：`deepmt/analysis/reporting/evidence_collector.py` 当前字段以单框架批次为核心。

**修复**：为 cross session 提供 `EvidencePack.kind = "cross_framework_divergence"` 分支，字段补 `framework2, framework2_version, f2_output_summary`，使 `deepmt case build --from-evidence` 能直接消费 cross 的产物。

### G6  oracle-遮蔽型数值差异无告警【中】

**现象**：`deepmt test cross exp` n=500 时 `numeric_diff` 多达 300+、`output_max_diff=5.4e8`，但 `consistency=100%`（两框架都 pass/fail 同结论），CLI 无任何醒目提示。Phase M 因此漏看 exp 的跨框架异常。

**根因**：cross 汇总只按"oracle 结论是否一致"给判定；数值差只在日志中。

**修复**：cross 汇总新增指标 `silent_numeric_diff_rate`（当 `output_close=False` 但 `consistency=1.0`），超过阈值时终端输出 `[!]` 告警；session JSON 持久化该指标。

### G7  `deepmt test cross --framework1 paddle` 未开放【中】

**现象**：当前 `--framework1` 仅用于选择主框架计算 oracle；默认且实际只用 pytorch。Paddle↔NumPy、Paddle↔TensorFlow 等方向测不了。

**修复**：放开 `--framework1` 对 paddlepaddle 的支持（依赖 G3）；增加 `deepmt test cross --matrix` 一次跑所有 `(f1,f2)` 对。

### G8  `deepmt test dedup` 无法消费 cross session【低】

**现象**：`dedup` 当前只聚类 `data/results/evidence/*.json`；cross 的 `data/results/cross_framework/*.json` 不在输入路径。

**修复**：`dedup` 接受 `--source cross|evidence|all`，以 `(operator, mr_id, framework_pair, diff_type)` 四元组聚类 cross 差异。

### G9  `deepmt mr generate` 需要 `OPENAI_API_KEY`【低-待观测】

**现象**：对未覆盖的算子（如 cos/sin）自动造 MR 依赖 LLM。离线环境或无 key 时此路径断掉，退回模板生成器 `--sources template`。

**修复**（非必须）：为无 LLM 场景补一套模板型 MR 种子（单调、奇偶、有界、零点、加法定理等），保证 `deepmt mr generate <op> --sources template --save` 至少产出 1 条可执行 MR；LLM 可选增强。

## 3. 修改计划（任务拆分）

| Task | 内容 | 估计改动面 | 依赖 |
|---|---|---|---|
| T1 | 扩展 `paddle_plugin._build_paddle_operators` 算子表（G1） | 1 个文件 | 无 |
| T2 | 扩展 `numpy_plugin._NUMPY_OPERATORS`（G2） | 1 个文件 | 无 |
| T3 | 放开 `batch` 对 paddle 的支持（G3） | `commands/test/execution/batch.py` + 若干门卫 | T1 |
| T4 | `CrossFrameworkTester` 保留失败样本（G4） | `analysis/qa/cross_framework_tester.py` | 无 |
| T5 | `EvidencePack.cross_framework` 变体（G5） | `evidence_collector.py` + `defect_case_builder.py` | T4 |
| T6 | cross 汇总加 silent-numeric-diff 告警（G6） | `cross_framework_tester.py` + CLI 输出 | 无 |
| T7 | cross `--framework1 paddle` 与 `--matrix`（G7） | CLI + tester | T1 |
| T8 | `dedup --source cross`（G8） | `defect_deduplicator.py` + CLI | T4 |
| T9 | 模板-only MR 生成降级保障（G9） | `mr_generator/base/*` | 无 |

建议执行顺序：**T1 → T2 → T4 → T6 → T3 → T5 → T8 → T7 → T9**。T9 可延后。

## 4. 验收标准（以"系统自主挖掘缺陷"为最终导向）

本阶段验收不以"找到真实缺陷"为门槛（历史教训：那是结果，不是工程目标），而以"CLI 自主闭环能完成多少步"为门槛。分三级：

### L1 —— CLI 到达能力（硬性，必须全绿）

- [ ] `deepmt test cross log_softmax --framework2 paddle --n-samples 50` 不再报 `无对应实现`，返回结构化一致性统计。
- [ ] `deepmt test cross gelu --framework2 numpy --n-samples 50` 不再 `unsupported_f2=50`，numpy 侧返回数值结果。
- [ ] `deepmt test batch --framework paddlepaddle --operator relu --n-samples 10 --collect-evidence` 正常跑通并落盘证据。
- [ ] `deepmt test cross <op> --framework2 paddle --save` 的 session JSON 中，当 `inconsistent_cases>0` 时必须包含 `failed_samples` 字段，且字段里能找到足以在独立 Python 复现问题的 input payload。

### L2 —— CLI 闭环能力（硬性，必须全绿）

- [ ] `deepmt test cross` 产出的跨框架不一致，能被 `deepmt case build --from-evidence <id>` 直接消费，生成含 `reproduce.py` 的案例包。
- [ ] `reproduce.py` 在干净 Python 环境内一次运行即复现原差异（与 session 中记录的 `diff_type` 一致）。
- [ ] `deepmt test dedup --source cross` 能对 session 中的不一致样本按 `(op, mr_id, framework_pair, diff_type)` 聚类输出 `DefectLead`。
- [ ] silent-numeric-diff 告警：`deepmt test cross exp --framework2 paddle` 必须在终端给出 `[!]` 级别提示，而不是沉默地 `consistency=100%`。

### L3 —— 自主挖掘能力（软性验收，用受控缺陷衡量）

不要求挖到真实框架缺陷；要求系统能在"受控注入"下自主完成"生成 MR → 发现差异 → 生成证据包 → 聚类 → 落盘案例"闭环。具体指标：

- [ ] 在 `FaultyPyTorchPlugin.BUILTIN_FAULT_CATALOG` 中新增至少 3 个**偏僻算子**的缺陷（cos/sin/sqrt 以外的，例如 cosh/tan/log1p/expm1），且：
    - 测试前：知识库无对应 MR、插件映射无特殊硬编码、cross 路径 default 参考框架有等价实现（依赖 T1/T2）；
    - 运行路径：`deepmt mr generate <op> --save` → `deepmt test open --operator <op> --inject-faults <op>:<mutant> --collect-evidence` → `deepmt test dedup` → `deepmt case build --from-evidence`；
    - 通过标准：每个受控缺陷都被自动生成的 MR（或模板回退 MR）捕获，至少一个相应的 `DefectLead` 产生，`case build` 产出可复现的 `reproduce.py`。
- [ ] 在同一 CLI 闭环下跑"无注入"对照（未激活 faulty 插件），**必须**零 `DefectLead`（False Positive 率验证）。

满足 L1+L2+L3 即认为系统自主挖掘能力具备，可在此基础上重启 Phase M 真实缺陷扫描。

## 5. 与上下游阶段的关系

- 上游 Phase L（实验基准）已固化；本阶段不动 `experiments/` 下的任何 RQ 口径。
- 下游 Phase N（论文交付）依赖本阶段 L2 给出的跨框架差异案例模板；若本阶段完成，Phase N 的"真实缺陷补充"子任务即可恢复按计划执行。
- 与 `phase_m_real_defect_hunting/02_candidate_cases.md` 的衔接：C1（Paddle log_softmax 截断）只有在 T1+T2+T4 完成后才能被 CLI 自主重发现，届时是本阶段 L1+L2 的天然回归用例。

## 6. 不在本阶段范围

- 不扩充 MR 知识库数量（L3 的注入实验以 `mr generate` 即时产出为准）。
- 不做真实缺陷挖掘（这是阶段目标，但验收不以此为门槛；见 §4 L3 软性验收）。
- 不改 MR 生成的四阶段流水线（operator/model/application 三层主逻辑）。
- 不引入 TensorFlow 插件（保持 `NotImplementedError`）。

## 7. 开发完成标志

- §4 所有 L1+L2 勾选项全部通过；
- §4 L3 中至少 3 个偏僻算子注入实验完成闭环；
- `docs/phase_m_real_defect_hunting/03_cli_only_sweep.md` §5 列出的四项 CLI 能力全部关闭；
- `docs/dev/status.md` 勾选本阶段完成，并链接回本文件。
