"""MR 重复检测（K4）。

MRDeduplicator 通过比较 oracle_expr 和 transform_code 的规范化形式，
识别知识库中的重复或高度相似的 MR。

策略：
    - oracle_expr 空格规范化后完全相同 → 强重复
    - oracle_expr 完全相同 + transform_code 高度相似 → 弱重复
    - 建议：保留 lifecycle_state 更高的那条
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from deepmt.ir.schema import MetamorphicRelation


@dataclass
class DuplicateGroup:
    """一组重复 MR（包含强/弱重复）。"""

    canonical_id: str                   # 推荐保留的 MR ID
    duplicate_ids: List[str] = field(default_factory=list)  # 建议归档的 MR ID
    similarity_type: str = "strong"     # "strong" | "weak"
    reason: str = ""


def _normalize_expr(expr: str) -> str:
    """规范化表达式：去除多余空格，统一操作符周围格式。"""
    expr = re.sub(r"\s+", " ", expr.strip())
    expr = re.sub(r"\s*==\s*", "==", expr)
    expr = re.sub(r"\s*\+\s*", "+", expr)
    expr = re.sub(r"\s*\*\s*", "*", expr)
    return expr


def _normalize_transform(code: str) -> str:
    """规范化 transform_code：去除空格，统一 lambda 格式。"""
    return re.sub(r"\s+", "", code.strip())


class MRDeduplicator:
    """MR 重复检测器。

    用法::

        dedup = MRDeduplicator()
        groups = dedup.find_duplicates(mrs)
        for g in groups:
            print(g.canonical_id, "→ retire:", g.duplicate_ids)
    """

    def find_duplicates(
        self, mrs: List["MetamorphicRelation"]
    ) -> List[DuplicateGroup]:
        """检测列表中的重复 MR，返回去重建议分组。

        策略：
            1. 完全相同的 oracle_expr（规范化后）→ 强重复
            2. 相同 oracle_expr 且 transform_code 相同 → 强重复（已涵盖）

        在每组重复中，按质量等级排序，保留最高质量的 MR 作为 canonical。
        """
        from collections import defaultdict
        from deepmt.mr_governance.quality import QualityLevel

        # group by normalized oracle_expr
        expr_groups: dict = defaultdict(list)
        for mr in mrs:
            key = _normalize_expr(mr.oracle_expr)
            expr_groups[key].append(mr)

        groups: List[DuplicateGroup] = []
        for key, group in expr_groups.items():
            if len(group) < 2:
                continue

            # 按质量等级排序，quality 高的排前面
            group_sorted = sorted(
                group,
                key=lambda m: QualityLevel.from_lifecycle(m.lifecycle_state),
                reverse=True,
            )
            canonical = group_sorted[0]
            duplicates = [m.id for m in group_sorted[1:]]

            groups.append(
                DuplicateGroup(
                    canonical_id=canonical.id,
                    duplicate_ids=duplicates,
                    similarity_type="strong",
                    reason=f"oracle_expr 相同: '{key}'",
                )
            )

        return groups

    def filter_unique(
        self, mrs: List["MetamorphicRelation"]
    ) -> Tuple[List["MetamorphicRelation"], List[DuplicateGroup]]:
        """去重并返回 (唯一MR列表, 重复分组列表)。

        唯一列表保留每组中质量最高的 MR，丢弃其余（不修改原对象）。
        """
        groups = self.find_duplicates(mrs)
        retire_ids: set = set()
        for g in groups:
            retire_ids.update(g.duplicate_ids)

        unique = [m for m in mrs if m.id not in retire_ids]
        return unique, groups
