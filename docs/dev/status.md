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
| Phase N：论文交付收口与复现资产封装           | ⬜ 未开始  |

**当前主链：** A~L 已完成 → **Phase M 进行中（M1~M7 已实现主体流程，2 个案例已归档）**

---

## 测试覆盖

**全部 718 个单元测试通过（无 LLM/网络依赖），另有 31 个集成测试通过。**（共 749 个测试；Phase M 新增 18 个集成测试）

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

*最后更新：2026-04-14（Phase M 主体流程实现完成；等待 Phase N）*
