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
| Phase K：全层MR质量保障与统一知识库治理       | ⬜ 未开始  |
| Phase L：论文实验基准与自动化数据生产线       | ⬜ 未开始  |
| Phase M：真实缺陷挖掘与案例沉淀               | ⬜ 未开始  |
| Phase N：论文交付收口与复现资产封装           | ⬜ 未开始  |

**当前主链：** A~J 已完成 → **下一步：全层MR质量保障与统一知识库治理（Phase K）**

---

## 测试覆盖

**全部 695 个单元测试通过（无 LLM/网络依赖），另有 13 个集成测试通过。**（共 708 个测试）

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

*最后更新：2026-04-13（Phase J 全部完成；下一步进入 Phase K）*
