"""
RQ1-RQ4 实验口径定义。

每个 RQ 对象描述：
  - 研究问题的完整表述
  - 核心指标及其计算来源
  - 数据来源模块
  - 聚合策略

用途：
  - 作为 ExperimentOrganizer 的配置根，确保指标计算口径在整个系统中唯一
  - 供导出脚本按 RQ 维度查询、过滤和格式化结果
  - 为论文表格和图表提供统一的列/标题命名

用法::

    from deepmt.experiments.rq_config import RQ_DEFINITIONS, get_rq

    cfg = get_rq("rq1")
    print(cfg.question)
    for m in cfg.metrics:
        print(m.name, m.description)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MetricSpec:
    """单个指标的规格描述。"""

    name: str
    """指标在输出 dict 中的 key（与 ExperimentOrganizer 保持一致）。"""

    description: str
    """人类可读的中文指标描述。"""

    unit: str = ""
    """单位（百分比、数量、比率等）。"""

    higher_is_better: Optional[bool] = None
    """True=越高越好, False=越低越好, None=中性/无方向。"""

    thesis_label: str = ""
    """论文表格中的列头（可含 LaTeX）。"""


@dataclass(frozen=True)
class RQConfig:
    """单个研究问题的完整实验口径定义。"""

    id: str
    """RQ 标识符，如 'rq1'。"""

    question: str
    """研究问题完整表述。"""

    data_source: str
    """主要数据来源模块说明。"""

    metrics: List[MetricSpec]
    """该 RQ 的全部核心指标（有序）。"""

    aggregate_strategy: str = ""
    """聚合策略说明（per-operator 均值、全局求和等）。"""

    notes: str = ""
    """额外说明（已知限制、手动填充字段等）。"""

    def metric_names(self) -> List[str]:
        return [m.name for m in self.metrics]

    def metric_by_name(self, name: str) -> Optional[MetricSpec]:
        for m in self.metrics:
            if m.name == name:
                return m
        return None


# ── RQ1：MR 自动生成质量 ─────────────────────────────────────────────────────

RQ1 = RQConfig(
    id="rq1",
    question="DeepMT 能否自动生成高质量的蜕变关系（MR）？生成的 MR 在数量、验证率和覆盖广度上表现如何？",
    data_source="MRRepository (data/knowledge/mr_repository/operator/*.yaml)",
    metrics=[
        MetricSpec(
            name="total_mr_count",
            description="知识库中 MR 总数",
            unit="条",
            higher_is_better=True,
            thesis_label="\\#MR",
        ),
        MetricSpec(
            name="verified_mr_count",
            description="已通过数值预检或 SymPy 证明的 MR 数量",
            unit="条",
            higher_is_better=True,
            thesis_label="\\#Verified",
        ),
        MetricSpec(
            name="verification_rate",
            description="验证率 = verified / total",
            unit="%",
            higher_is_better=True,
            thesis_label="Verify Rate",
        ),
        MetricSpec(
            name="operators_with_mr",
            description="至少有一条 MR 的算子数",
            unit="个",
            higher_is_better=True,
            thesis_label="\\#Operators",
        ),
        MetricSpec(
            name="avg_mr_per_operator",
            description="平均每算子 MR 数",
            unit="条/算子",
            higher_is_better=True,
            thesis_label="Avg MR/Op",
        ),
        MetricSpec(
            name="category_distribution",
            description="各 MR 分类的数量分布（dict）",
            unit="",
            higher_is_better=None,
            thesis_label="Category Dist.",
        ),
        MetricSpec(
            name="source_distribution",
            description="MR 来源分布（llm / template / manual）",
            unit="",
            higher_is_better=None,
            thesis_label="Source Dist.",
        ),
    ],
    aggregate_strategy="全局统计（跨所有算子汇总）",
)

# ── RQ2：缺陷检测能力 ─────────────────────────────────────────────────────────

RQ2 = RQConfig(
    id="rq2",
    question="DeepMT 能否有效检测深度学习框架中的缺陷？在标准 benchmark 上的失败率和可复现证据如何？",
    data_source="ResultsManager (data/results/defects.db) + EvidenceCollector (data/results/evidence/)",
    metrics=[
        MetricSpec(
            name="total_test_cases",
            description="总测试用例数（输入×MR 组合数）",
            unit="条",
            higher_is_better=True,
            thesis_label="\\#Tests",
        ),
        MetricSpec(
            name="total_failed",
            description="蜕变关系被违反（失败）的用例数",
            unit="条",
            higher_is_better=None,
            thesis_label="\\#Failed",
        ),
        MetricSpec(
            name="overall_pass_rate",
            description="整体通过率",
            unit="%",
            higher_is_better=None,
            thesis_label="Pass Rate",
        ),
        MetricSpec(
            name="failure_rate",
            description="失败率（oracle 违反率）",
            unit="%",
            higher_is_better=None,
            thesis_label="Fail Rate",
        ),
        MetricSpec(
            name="operators_tested",
            description="被测算子总数",
            unit="个",
            higher_is_better=True,
            thesis_label="\\#Ops Tested",
        ),
        MetricSpec(
            name="operators_with_failure",
            description="存在至少一次失败的算子数",
            unit="个",
            higher_is_better=None,
            thesis_label="\\#Ops w/ Fail",
        ),
        MetricSpec(
            name="evidence_pack_count",
            description="可复现证据包数量（每个对应一个潜在缺陷）",
            unit="个",
            higher_is_better=True,
            thesis_label="\\#Evidence",
        ),
        MetricSpec(
            name="unique_defect_leads",
            description="去重后独立缺陷线索数",
            unit="条",
            higher_is_better=True,
            thesis_label="\\#Unique Defects",
        ),
    ],
    aggregate_strategy="全局统计（跨所有测试会话汇总）",
)

# ── RQ3：跨框架一致性 ─────────────────────────────────────────────────────────

RQ3 = RQConfig(
    id="rq3",
    question="DeepMT 能否有效发现深度学习框架间的一致性差异？一致率和差异量级如何？",
    data_source="CrossFrameworkTester (data/results/cross_framework/*.json)",
    metrics=[
        MetricSpec(
            name="cross_session_count",
            description="已完成的跨框架对比实验数",
            unit="次",
            higher_is_better=True,
            thesis_label="\\#Sessions",
        ),
        MetricSpec(
            name="operators_compared",
            description="参与跨框架对比的算子数",
            unit="个",
            higher_is_better=True,
            thesis_label="\\#Ops",
        ),
        MetricSpec(
            name="overall_consistency_rate",
            description="所有实验的平均一致率",
            unit="%",
            higher_is_better=True,
            thesis_label="Consistency",
        ),
        MetricSpec(
            name="avg_output_max_diff",
            description="平均输出最大差值（体现实现差异量级）",
            unit="float",
            higher_is_better=False,
            thesis_label="Avg MaxDiff",
        ),
        MetricSpec(
            name="inconsistent_mr_count",
            description="至少存在一个不一致样本的 MR 数量",
            unit="条",
            higher_is_better=None,
            thesis_label="\\#Inconsistent MR",
        ),
        MetricSpec(
            name="framework_pairs",
            description="参与对比的框架对列表",
            unit="",
            higher_is_better=None,
            thesis_label="Framework Pairs",
        ),
    ],
    aggregate_strategy="跨框架对均值；逐 session 明细另行列出",
)

# ── RQ4：与基线对比 ───────────────────────────────────────────────────────────

RQ4 = RQConfig(
    id="rq4",
    question="与人工/基线方法相比，DeepMT 在覆盖度、自动化程度和用例密度上的优势如何？",
    data_source="综合 RQ1+RQ2 统计，基线数值需人工填写",
    metrics=[
        MetricSpec(
            name="operators_covered",
            description="DeepMT 覆盖的算子总数（有 MR 或有测试结果）",
            unit="个",
            higher_is_better=True,
            thesis_label="\\#Ops Covered",
        ),
        MetricSpec(
            name="avg_mrs_per_operator",
            description="平均每算子 MR 数（衡量用例密度）",
            unit="条/算子",
            higher_is_better=True,
            thesis_label="MR Density",
        ),
        MetricSpec(
            name="test_density",
            description="平均每算子测试用例数",
            unit="条/算子",
            higher_is_better=True,
            thesis_label="Test Density",
        ),
        MetricSpec(
            name="automation_scope",
            description="自动化链路范围说明（文字）",
            unit="",
            higher_is_better=None,
            thesis_label="Automation Scope",
        ),
    ],
    aggregate_strategy="引用 RQ1/RQ2 结果，基线数值单独列表（需人工填入）",
    notes=(
        "与基线对比的绝对数值（如 MTR、语句覆盖率）需手动填入基线方法的测量结果，"
        "DeepMT 侧数据已在 RQ1/RQ2/RQ3 中自动提供。"
    ),
)

# ── 注册表 ────────────────────────────────────────────────────────────────────

RQ_DEFINITIONS: Dict[str, RQConfig] = {
    "rq1": RQ1,
    "rq2": RQ2,
    "rq3": RQ3,
    "rq4": RQ4,
}


def get_rq(rq_id: str) -> Optional[RQConfig]:
    """按 id 获取 RQConfig，未找到返回 None。"""
    return RQ_DEFINITIONS.get(rq_id.lower())


def list_rqs() -> List[RQConfig]:
    """返回所有 RQConfig（按 rq1→rq4 顺序）。"""
    return [RQ_DEFINITIONS[k] for k in ("rq1", "rq2", "rq3", "rq4")]
