"""MR 用户工作区仓库：以 YAML 文件持久化存储蜕变关系（每算子一文件）。

存储结构：
    data/knowledge/mr_repository/operator/<operator_name>.yaml

每个文件格式：
    operator: torch.add
    generated_at: "2026-01-01T00:00:00"
    mrs:
      - id: "abc-123"
        description: "..."
        transform_code: "lambda k: {**k, 'input': 2.0 * k['input']}"
        oracle_expr: "trans == 2.0 * orig"
        category: "linearity"
        tolerance: 1.0e-6
        source: "llm"
        applicable_frameworks: ["pytorch"]
        checked: true
        proven: true
        verified: true
        analysis: "..."
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation


class MRRepository:
    """用户工作区 MR 仓库：每个算子的 MR 列表存为独立 YAML 文件。"""

    def __init__(self, repo_dir: str = "data/knowledge/mr_repository/operator"):
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📦 [REPO] MR repository at: {self.repo_dir}")

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _op_file(self, operator_name: str) -> Path:
        safe_name = operator_name.replace("/", "__")
        return self.repo_dir / f"{safe_name}.yaml"

    def _serialize_mr(self, mr: MetamorphicRelation) -> Dict:
        d: Dict = {
            "id": mr.id,
            "description": mr.description,
            "transform_code": mr.transform_code,
            "oracle_expr": mr.oracle_expr,
            "category": mr.category,
            "tolerance": mr.tolerance,
            "layer": mr.layer,
            "source": mr.source,
            "applicable_frameworks": mr.applicable_frameworks,
            "checked": mr.checked,
            "proven": mr.proven,
            "verified": mr.verified,
            "analysis": mr.analysis,
        }
        return d

    def _deserialize_mr(self, data: Dict) -> MetamorphicRelation:
        transform_code = data.get("transform_code", "")
        try:
            transform = eval(transform_code) if transform_code else lambda *args: args
        except Exception:
            transform = lambda *args: args

        return MetamorphicRelation(
            id=data["id"],
            description=data.get("description", ""),
            transform_code=transform_code,
            transform=transform,
            oracle_expr=data.get("oracle_expr", ""),
            category=data.get("category", "general"),
            tolerance=float(data.get("tolerance", 1e-6)),
            layer=data.get("layer", "operator"),
            source=data.get("source", ""),
            applicable_frameworks=data.get("applicable_frameworks"),
            checked=data.get("checked"),
            proven=data.get("proven"),
            verified=data.get("verified", False),
            analysis=data.get("analysis", ""),
        )

    def _load_file(self, operator_name: str) -> Optional[Dict]:
        path = self._op_file(operator_name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write_file(self, operator_name: str, data: Dict):
        path = self._op_file(operator_name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def save(
        self,
        operator_name: str,
        mrs: List[MetamorphicRelation],
        framework: Optional[str] = None,
    ) -> int:
        """将 MR 列表写入该算子的 YAML 文件（覆盖已有内容）。

        framework 为便利参数：若 MR 的 applicable_frameworks 为空则自动填入。
        """
        if not mrs:
            return 0

        for mr in mrs:
            if framework and not mr.applicable_frameworks:
                mr.applicable_frameworks = [framework]

        data = {
            "operator": operator_name,
            "generated_at": datetime.now().isoformat(),
            "mrs": [self._serialize_mr(mr) for mr in mrs],
        }
        self._write_file(operator_name, data)
        logger.info(f"📦 [REPO] Saved {len(mrs)} MRs for operator: {operator_name}")
        return len(mrs)

    def load(
        self,
        operator_name: str,
        framework: Optional[str] = None,
    ) -> List[MetamorphicRelation]:
        """从 YAML 文件加载 MR 列表，可按 applicable_frameworks 过滤。"""
        data = self._load_file(operator_name)
        if not data:
            return []

        mrs = []
        for entry in data.get("mrs", []):
            try:
                mr = self._deserialize_mr(entry)
                if framework and mr.applicable_frameworks is not None:
                    if framework not in mr.applicable_frameworks:
                        continue
                mrs.append(mr)
            except Exception as e:
                logger.error(f"Error loading MR: {e}")

        logger.info(f"📦 [REPO] Loaded {len(mrs)} MRs for operator: {operator_name}")
        return mrs

    def exists(self, operator_name: str) -> bool:
        return self._op_file(operator_name).exists()

    def list_operators(self) -> List[str]:
        return sorted(f.stem for f in self.repo_dir.glob("*.yaml"))

    def delete(
        self,
        operator_name: str,
        mr_id: Optional[str] = None,
    ) -> int:
        """删除 MR 记录。mr_id=None 则删除整个算子文件，否则仅删除指定条目。"""
        path = self._op_file(operator_name)
        if not path.exists():
            return 0

        if mr_id is None:
            data = self._load_file(operator_name)
            count = len(data.get("mrs", [])) if data else 0
            path.unlink()
            logger.info(f"📦 [REPO] Deleted all {count} MRs for '{operator_name}'")
            return count

        data = self._load_file(operator_name)
        if not data:
            return 0
        before = len(data.get("mrs", []))
        data["mrs"] = [m for m in data.get("mrs", []) if m.get("id") != mr_id]
        deleted = before - len(data["mrs"])
        if deleted > 0:
            self._write_file(operator_name, data)
        logger.info(f"📦 [REPO] Deleted {deleted} MR(s) for '{operator_name}' mr_id={mr_id}")
        return deleted

    def list_operators_by_framework(self, framework: str) -> List[str]:
        """列出包含指定框架 MR 的算子（applicable_frameworks 包含该框架，或为 None）。"""
        result = []
        for op in self.list_operators():
            for mr in self.load(op):
                if mr.applicable_frameworks is None or framework in mr.applicable_frameworks:
                    result.append(op)
                    break
        return sorted(result)

    def get_mr_with_validation_status(
        self,
        operator_name: str,
        verified_only: bool = False,
        framework: Optional[str] = None,
    ) -> List[MetamorphicRelation]:
        """加载 MR，支持 verified_only 过滤。"""
        mrs = self.load(operator_name, framework=framework)
        if verified_only:
            mrs = [m for m in mrs if m.verified]
        return mrs

    def get_statistics(self, operator_name: Optional[str] = None) -> Dict:
        """统计 MR 数量信息（total/verified/unverified/checked/proven）。"""

        def _stats_for(mrs: List[MetamorphicRelation]) -> Dict:
            return {
                "total": len(mrs),
                "verified": sum(1 for m in mrs if m.verified),
                "unverified": sum(1 for m in mrs if not m.verified),
                "checked": sum(1 for m in mrs if m.checked is True),
                "proven": sum(1 for m in mrs if m.proven is True),
            }

        if operator_name:
            c = _stats_for(self.load(operator_name))
            return {
                "total_mrs": c["total"],
                "verified_mrs": c["verified"],
                "unverified_mrs": c["unverified"],
                "checked": c["checked"],
                "proven": c["proven"],
                "by_operator": {operator_name: c},
            }

        stats: Dict = {
            "total_mrs": 0,
            "verified_mrs": 0,
            "unverified_mrs": 0,
            "checked": 0,
            "proven": 0,
            "by_operator": {},
        }
        for op in self.list_operators():
            c = _stats_for(self.load(op))
            stats["total_mrs"] += c["total"]
            stats["verified_mrs"] += c["verified"]
            stats["unverified_mrs"] += c["unverified"]
            stats["checked"] += c["checked"]
            stats["proven"] += c["proven"]
            stats["by_operator"][op] = c
        return stats
