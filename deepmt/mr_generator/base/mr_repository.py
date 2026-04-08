"""MR 仓库：以 YAML 文件持久化存储蜕变关系（每算子一文件）。

存储结构：
    data/mr_repository/<operator_name>.yaml

每个文件格式：
    operator: torch.nn.functional.relu
    framework: pytorch
    generated_at: "2024-01-01T00:00:00"
    mrs:
      - id: "abc-123"
        description: "..."
        transform_code: "lambda k: {**k, 'input': 2.0 * k['input']}"
        oracle_expr: "trans == 2.0 * orig"
        category: "linearity"
        tolerance: 1.0e-6
        verified: true
        precheck_passed: null
        sympy_proven: null
        created_at: "2024-01-01T00:00:00"
        applicable_frameworks:
          - pytorch
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation


class MRRepository:
    """MR 仓库：每个算子的 MR 列表存为独立 YAML 文件，便于人工查阅与版本追踪。"""

    def __init__(self, repo_dir: str = "data/mr_repository"):
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📦 [REPO] MR repository at: {self.repo_dir}")

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _op_file(self, operator_name: str) -> Path:
        """算子名 → YAML 文件路径（将 / 替换为 __ 以确保路径安全）。"""
        safe_name = operator_name.replace("/", "__")
        return self.repo_dir / f"{safe_name}.yaml"

    def _serialize_mr(self, mr: MetamorphicRelation) -> Dict:
        return {
            "id": mr.id,
            "description": mr.description,
            "transform_code": mr.transform_code,
            "oracle_expr": mr.oracle_expr,
            "category": mr.category,
            "tolerance": mr.tolerance,
            "analysis": mr.analysis,
            "layer": mr.layer,
            "verified": mr.verified,
            "precheck_passed": getattr(mr, "_precheck_passed", None),
            "sympy_proven": getattr(mr, "_sympy_proven", None),
            "created_at": datetime.now().isoformat(),
            "applicable_frameworks": mr.applicable_frameworks,
        }

    def _deserialize_mr(self, data: Dict) -> MetamorphicRelation:
        transform_code = data.get("transform_code", "")
        try:
            transform = eval(transform_code) if transform_code else lambda *args: args
        except Exception:
            transform = lambda *args: args

        mr = MetamorphicRelation(
            id=data["id"],
            description=data.get("description", ""),
            transform=transform,
            transform_code=transform_code,
            oracle_expr=data.get("oracle_expr", ""),
            category=data.get("category", "general"),
            tolerance=float(data.get("tolerance", 1e-6)),
            analysis=data.get("analysis", ""),
            layer=data.get("layer", "operator"),
            verified=data.get("verified", False),
            applicable_frameworks=data.get("applicable_frameworks"),
        )
        # 附加验证字段（仅供统计使用，不在 schema 中定义）
        mr._precheck_passed = data.get("precheck_passed")  # type: ignore[attr-defined]
        mr._sympy_proven = data.get("sympy_proven")  # type: ignore[attr-defined]
        return mr

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
        """将 MR 列表写入该算子的 YAML 文件（覆盖已有内容）。"""
        if not mrs:
            return 0

        for mr in mrs:
            if framework and not mr.applicable_frameworks:
                mr.applicable_frameworks = [framework]

        data = {
            "operator": operator_name,
            "framework": framework,
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
        """从 YAML 文件加载 MR 列表，可按框架过滤。"""
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
        """检查算子是否有保存的 MR 文件。"""
        return self._op_file(operator_name).exists()

    def list_operators(self) -> List[str]:
        """列出所有有 MR 的算子名称（排除 mr_templates.yaml）。"""
        return sorted(
            f.stem
            for f in self.repo_dir.glob("*.yaml")
            if f.name != "mr_templates.yaml"
        )

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
        """加载 MR，支持 verified_only 过滤（与 load() 等价，供 CLI 调用）。"""
        mrs = self.load(operator_name, framework=framework)
        if verified_only:
            mrs = [m for m in mrs if m.verified]
        return mrs

    def get_statistics(self, operator_name: Optional[str] = None) -> Dict:
        """统计 MR 数量信息（total/verified/unverified/precheck_passed/sympy_proven）。"""

        def _stats_for(mrs: List[MetamorphicRelation]) -> Dict:
            return {
                "total": len(mrs),
                "verified": sum(1 for m in mrs if m.verified),
                "unverified": sum(1 for m in mrs if not m.verified),
                "precheck_passed": sum(
                    1 for m in mrs if getattr(m, "_precheck_passed", None) is True
                ),
                "sympy_proven": sum(
                    1 for m in mrs if getattr(m, "_sympy_proven", None) is True
                ),
            }

        if operator_name:
            c = _stats_for(self.load(operator_name))
            return {
                "total_mrs": c["total"],
                "verified_mrs": c["verified"],
                "unverified_mrs": c["unverified"],
                "precheck_passed": c["precheck_passed"],
                "sympy_proven": c["sympy_proven"],
                "by_operator": {operator_name: c},
            }

        stats: Dict = {
            "total_mrs": 0,
            "verified_mrs": 0,
            "unverified_mrs": 0,
            "precheck_passed": 0,
            "sympy_proven": 0,
            "by_operator": {},
        }
        for op in self.list_operators():
            c = _stats_for(self.load(op))
            stats["total_mrs"] += c["total"]
            stats["verified_mrs"] += c["verified"]
            stats["unverified_mrs"] += c["unverified"]
            stats["precheck_passed"] += c["precheck_passed"]
            stats["sympy_proven"] += c["sympy_proven"]
            stats["by_operator"][op] = c
        return stats
