# DeepMT 开发状态

## 阶段进度

| 阶段                                          | 状态      |
| --------------------------------------------- | --------- |
| Phase A：算子数据层完善（A1~A6）              | ✅ 完成    |
| Phase B：算子层 MR 生成与知识库（B1~B3）      | ✅ 完成    |
| Phase C：测试执行与跨框架适配                 | ✅ 完成    |
| Phase D：缺陷分析与实验闭环                   | ✅ 完成    |
| Phase E：演示交付与生产化加固（E1~E6）        | ✅ 完成    |
| Phase F：软件工程规范化与包发布准备（F1~F11） | ✅ 完成    |
| Phase G：统一IR与三层对象建模                 | ✅ 完成    |
| Phase H：第二框架落地与真实跨框架适配         | ✅ 完成    |
| Phase I：模型层MR自动生成引擎                 | ✅ 完成    |
| Phase J：应用层语义MR生成与验证               | ✅ 完成    |
| Phase K：全层MR质量保障与统一知识库治理       | ✅ 完成    |
| Phase L：论文实验基准与自动化数据生产线       | ✅ 完成    |
| Phase M：真实缺陷挖掘与案例沉淀               | 🔄 进行中  |
| Phase M 系统能力缺口修复（前置 T1~T9）        | ✅ 完成    |
| Phase N：论文交付收口与复现资产封装           | ⬜ 未开始  |

**当前主链：** A~L 已完成 → **Phase M 系统能力缺口修复完成（L1+L2+L3 全绿，详见 [15_Phase_M_system_capability_gaps.md](./15_Phase_M_system_capability_gaps.md)）** → Phase M 主干真实缺陷扫描可重启

---

## 测试覆盖

**全部 739 个单元测试通过（无 LLM/网络依赖），另有 31 个集成测试通过。**（Phase M 系统能力缺口修复新增测试覆盖 T1~T9）

---

## 架构设计约定

- **框架参数化**：框架名称以 `FrameworkType` 参数传入，PyTorch 先行实现，其他框架入口抛 `NotImplementedError`。
- **MR 框架无关**：`MetamorphicRelation` 通过 `applicable_frameworks` 声明适用范围；`transform_code` 仅用 Python 原生算术；框架 tensor 包装/解包由插件负责。
- **算子双态**：`function`（`torch.nn.functional.relu`）与 `module`（`torch.nn.ReLU`）；当前算子层测试统一用 `function` 形态。
- **input_specs 质量分层**：`confirmed` / `auto-usable` / `weak`，通过 `input_specs_auto` 字段区分。

---

## 已知限制

1. **SymPy 验证限制**：含浮点的复杂性质无法符号证明，仅依赖 pre-check 数值验证
2. **LLM 依赖**：MR 猜想质量依赖提示工程与 API 密钥，单元测试通过 `use_llm=False` 隔离
3. **transform_code 可移植性**：跨框架 MR 要求 `transform_code` 不使用框架特定 API；PyTorch 阶段暂不强制

---

### Phase M 已完成模块（2026-04-14）

| 模块 | 路径 | 说明 |
|------|------|------|
| DefectCaseBuilder | `deepmt/analysis/defect_case_builder.py` | 缺陷线索 → 案例包自动构建器 |
| case CLI 命令组 | `deepmt/commands/case.py` | list/show/confirm/build/export 五个子命令 |
| 案例包 009eb89bcb | `deepmt/cases/real_defects/009eb89bcb/` | gelu MR 质量案例（confirmed） |
| 案例包 e861263744 | `deepmt/cases/real_defects/e861263744/` | exp float32 溢出边界案例（confirmed） |
| 缺陷挖掘流程文档 | `docs/phase_m_defect_hunting_process.md` | 完整流程说明与扩展建议 |
| 缺陷报告模板 | `deepmt/templates/bug_report_template.md` | 向外部社区提交缺陷报告用 |
| 集成测试 | `tests/integration/test_real_case_pipeline.py` | 18 个测试覆盖完整案例流水线 |
| MR YAML 扩展 | `data/knowledge/mr_repository/operator/` | 新增 gelu/tanh/leaky_relu/softmax/log_softmax |

---

### Phase M 系统能力缺口修复（2026-04-15）

| 任务 | 内容 | 主要改动 |
|------|------|---------|
| T1 | Paddle 插件算子表扩面 | `deepmt/plugins/paddle_plugin.py`（新增 log_softmax/layer_norm/logsumexp/tan/sinh/cosh 等 19 个算子） |
| T2 | NumPy 插件算子表扩面 | `deepmt/plugins/numpy_plugin.py`（新增 gelu/erf/log_softmax/layer_norm/logsumexp 等，无 scipy 依赖） |
| T3 | `test batch` 放开 paddlepaddle | `deepmt/commands/test/_group.py` `_SUPPORTED_FRAMEWORKS` |
| T4 | cross session 保留失败样本 | `CrossConsistencyResult.failed_samples` + `keep_samples` CLI 选项 |
| T5 | EvidencePack 跨框架变体 | `evidence_collector.create_cross / import_from_cross_session`、`_generate_cross_reproduce_script` |
| T6 | silent-numeric-diff 告警 | `CrossSessionResult.silent_numeric_diff_rate` + 终端 `[!]` 提示 |
| T7 | `test cross --matrix` | 三对 (pytorch,numpy)/(pytorch,paddle)/(paddle,numpy) 一次跑 |
| T8 | `test dedup --source cross` | `DefectDeduplicator._collect_cross_packs` 按 `(op, mr_id, fw_pair, bucket)` 聚类 |
| T9 | 模板-only MR 生成降级 | `MRTemplatePool.generate_mr_candidates` 无映射时回退 `discover_all_templates`，并新增短名兜底 |

验收记录：
- [L1 验收](../phase_m_real_defect_hunting/05_acceptance_L1.md) — CLI 到达能力 4/4
- [L2 验收](../phase_m_real_defect_hunting/06_acceptance_L2.md) — CLI 闭环能力 4/4
- [L3 验收](../phase_m_real_defect_hunting/07_acceptance_L3.md) — 受控缺陷自主挖掘闭环（cosh/asinh/expm1）+ 零 FP 对照

---

*最后更新：2026-04-15（Phase M 系统能力缺口 T1~T9 修复完成，L1+L2+L3 全绿；Phase M 主干真实缺陷扫描待重启）*
