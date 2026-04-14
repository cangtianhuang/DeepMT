# Bug Report Template — DeepMT 缺陷报告模板

> 用途：向框架社区（PyTorch Issues、TensorFlow GitHub 等）提交缺陷报告时使用本模板。  
> 填写说明：`[TODO: ...]` 表示需手工填写的字段；其余字段在构建案例包时自动生成。

---

## 标题（Title）

`[TODO: 简明描述缺陷，如 "torch.nn.functional.X: unexpected behavior when Y"]`

---

## 框架版本（Framework Version）

| 字段 | 值 |
|------|----|
| 框架 | `[framework]` |
| 版本 | `[framework_version]` |
| Python 版本 | `[TODO: python --version]` |
| 操作系统 | `[TODO: uname -a]` |
| CUDA 版本 | `[TODO: nvidia-smi 或 N/A]` |
| 安装方式 | `[TODO: pip / conda]` |

---

## 问题描述（Description）

`[TODO: 1-3 句话描述现象，如 "torch.exp() MR 在 float32 输入 x=88 时返回不对称 Inf"]`

---

## 最小复现脚本（Minimal Reproduction Script）

> 来源：`deepmt/cases/real_defects/[case_id]/reproduce.py`

```python
# [TODO: 粘贴 reproduce.py 内容，删除 DeepMT 导入部分，保留最小可运行代码]
import torch

# 输入设置
# ...

# 触发现象
# ...

# 预期 vs 实际
# expected: ...
# actual:   ...
```

---

## 预期行为（Expected Behavior）

`[TODO: 根据官方文档或数学定义描述预期输出]`

---

## 实际行为（Actual Behavior）

`[TODO: 描述实际观测到的异常输出]`

---

## 根因分析（Root Cause Analysis）

> 来源：CaseStudy.root_cause 字段

`[root_cause]`

---

## 发现方式（How Found）

本缺陷通过 DeepMT 蜕变关系测试系统自动发现：

- **MR ID**：`[mr_id]`
- **MR 描述**：`[mr_description]`
- **测试层次**：`[layer]`（算子层 / 模型层 / 应用层）
- **证据包 ID**：`[evidence_pack_path]`

DeepMT 通过以下蜕变关系发现该行为：

```
变换：[TODO: 描述输入变换]
Oracle：[TODO: 描述 oracle 断言]
违反情况：[TODO: 描述具体违反]
```

---

## 影响范围（Impact）

| 字段 | 值 |
|------|----|
| 严重程度 | `[severity]`（critical / high / medium / low） |
| 缺陷类型 | `[defect_type]` |
| 受影响版本 | `[affected_versions]` |
| 是否可复现 | ✅ 可稳定复现 |

---

## 交叉验证（Cross-validation）

`[TODO: 如已进行跨框架对比，在此填写对比结果]`

```
PyTorch [version]: [result]
NumPy   [version]: [result]
TensorFlow [version]: [result]
```

---

## 附件（Attachments）

- [ ] `reproduce.py` — 最小复现脚本
- [ ] `evidence.json` — 原始证据包（选填，含敏感系统信息时删除）
- [ ] 运行日志截图

---

## 案例归档信息（Internal）

> 此节为 DeepMT 内部归档信息，提交到外部 issue tracker 时可删除。

| 字段 | 值 |
|------|----|
| Case ID | `[case_id]` |
| 案例状态 | `[status]` |
| 创建时间 | `[created_at]` |
| 案例包路径 | `deepmt/cases/real_defects/[case_id]/` |

```bash
# 查看案例详情
deepmt case show [case_id]

# 更新案例状态（提交后）
deepmt case confirm [case_id] --status closed --notes "已提交至 upstream issue #XXXX"
```
