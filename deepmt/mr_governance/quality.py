"""统一质量级别定义（K2）。

QualityLevel 描述一条 MR 的可信程度，从低到高分五级：
    CANDIDATE — 已录入，未经任何验证（lifecycle_state=pending）
    CHECKED   — 数值 precheck 通过（lifecycle_state=checked）
    PROVEN    — SymPy 符号证明通过或人工确认（lifecycle_state=proven）
    CURATED   — 经显式人工策展，最高信任（lifecycle_state=curated）
    RETIRED   — 已废弃，不参与测试（lifecycle_state=retired）

质量分数（quality_score）用于排序和过滤：
    RETIRED=0, CANDIDATE=1, CHECKED=2, PROVEN=3, CURATED=4
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepmt.ir.schema import MetamorphicRelation


class QualityLevel(IntEnum):
    """MR 质量等级（数值越大越可信）。"""

    RETIRED = 0
    CANDIDATE = 1
    CHECKED = 2
    PROVEN = 3
    CURATED = 4

    @classmethod
    def from_lifecycle(cls, lifecycle_state: str) -> "QualityLevel":
        """从 lifecycle_state 字符串转换为 QualityLevel。"""
        _map = {
            "retired": cls.RETIRED,
            "pending": cls.CANDIDATE,
            "checked": cls.CHECKED,
            "proven": cls.PROVEN,
            "curated": cls.CURATED,
        }
        return _map.get(lifecycle_state, cls.CANDIDATE)

    @property
    def label(self) -> str:
        """返回小写字符串标签，与 lifecycle_state 兼容。"""
        _labels = {
            QualityLevel.RETIRED: "retired",
            QualityLevel.CANDIDATE: "candidate",
            QualityLevel.CHECKED: "checked",
            QualityLevel.PROVEN: "proven",
            QualityLevel.CURATED: "curated",
        }
        return _labels[self]

    def __str__(self) -> str:
        return self.label


def quality_level_from_mr(mr: "MetamorphicRelation") -> QualityLevel:
    """从 MetamorphicRelation 对象计算质量等级。"""
    return QualityLevel.from_lifecycle(mr.lifecycle_state)


def filter_by_quality(
    mrs: list,
    min_quality: QualityLevel = QualityLevel.CANDIDATE,
    exclude_retired: bool = True,
) -> list:
    """按质量等级过滤 MR 列表。

    Args:
        mrs:            MetamorphicRelation 列表
        min_quality:    最低质量要求（含）
        exclude_retired: 是否排除已归档关系（默认排除）

    Returns:
        满足条件的 MR 列表
    """
    result = []
    for mr in mrs:
        ql = quality_level_from_mr(mr)
        if exclude_retired and ql == QualityLevel.RETIRED:
            continue
        if ql >= min_quality:
            result.append(mr)
    return result


# 默认实验入库门槛：数值检查已通过（CHECKED 及以上）
DEFAULT_EXPERIMENT_MIN_QUALITY = QualityLevel.CHECKED
