# L3 验收报告（受控缺陷下的自主挖掘闭环）

日期: 2026-04-15

## 新增偏僻算子缺陷

在 `deepmt/plugins/faulty_pytorch_plugin.py::BUILTIN_FAULT_CATALOG` 中新增 3 个偏僻算子缺陷（均为原知识库零 MR、原目录未收录）：

| 算子 | 变异类型 | 破坏目标 |
| --- | --- | --- |
| `torch.cosh` | `identity` | 破坏 `cosh(-x)==cosh(x)` 偶函数性质 |
| `torch.asinh` | `negate` | 破坏 `asinh` 全局单调递增 |
| `torch.expm1` | `negate` | 破坏 `expm1` 全局单调递增 |

> 注：曾尝试 `torch.log1p + negate`，但预检随机输入偶尔落入 `x<=-1` 区间导致 log1p 输出 NaN，故替换为 `torch.asinh`（定义域全实数）。

## CLI 闭环：mr generate → test open → dedup → case build

对每个偏僻算子依次执行全流程，结果汇总：

| 算子 | mr generate（template-only） | test open 检出 | dedup DefectLead | case build | reproduce.py |
| --- | --- | --- | --- | --- | --- |
| `torch.cosh` | 1 MR（abs_even_symmetry，候选未验证） | 20/20 失败 `64af00e6-d22` | `0a2071a57273 NUMERICAL ×1` | `fc036b860c` | 运行通过，trans≠orig |
| `torch.asinh` | 1 MR（sigmoid_monotone_scale） | 20/20 失败 `07a3f4b8-3be` | `abfef0078525 OTHER ×1` | `b382388f68` | 运行通过，`INEQUALITY_VIOLATION 16/16` |
| `torch.expm1` | 1 MR（sigmoid_monotone_scale） | 20/20 失败 `0fd96bd2-d50` | `513529730446 OTHER ×1` | `46529a4bcc` | 运行通过，`INEQUALITY_VIOLATION 14/16` |

**结果**: ✅ 三个偏僻算子全部跑通，每个均产生 DefectLead 并生成可执行 reproduce.py。

## 零 FP 对照（无缺陷注入）

`deepmt test batch --operator <op> --n-samples 20 --collect-evidence`（使用默认 `PyTorchPlugin`，无 fault 注入）：

```
  [PASS] torch.cosh   MR=1  samples=20  pass=20/20  err=0
  [PASS] torch.asinh  MR=1  samples=20  pass=20/20  err=0
  [PASS] torch.expm1  MR=1  samples=20  pass=20/20  err=0
```

**结果**: ✅ 三算子均 0/20 失败；无证据包落盘，dedup 不会产生任何 DefectLead，False Positive 率 = 0。

## 过程中触及的系统改动（L3 暴露的小缺陷，顺手修复）

1. `MRTemplatePool.get_applicable_templates` 新增短名后缀兜底：`torch.cosh` 未命中全名时回退到 `cosh` 短名映射。
2. `data/knowledge/mr_repository/mr_templates.yaml::operator_mr_mapping` 新增 `cosh / asinh / expm1` → 模板映射条目。
3. `defect_case_builder._get_reproduce_script` 为 `kind=cross_framework_divergence` 证据包走 `_generate_cross_reproduce_script` 分支（L2 过程中已修，此处回归确认）。

## 小结

L3 全部通过。系统在受控注入下可以自主完成：MR 自动生成（模板回退）→ 批量开放测试发现失败 → 证据包落盘 → 去重产生 DefectLead → 构建案例包并生成可直接运行的 `reproduce.py`；同时无注入对照保持零 FP。

至此 L1 + L2 + L3 全绿，满足 `docs/dev/15_Phase_M_system_capability_gaps.md §7` 定义的系统能力缺口阶段完成标志（除 `docs/dev/status.md` 勾选与 `03_cli_only_sweep.md §5` 条目关闭两项待外部文档同步）。
