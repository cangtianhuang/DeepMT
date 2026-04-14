"""MR 用户工作区仓库：以 YAML 文件持久化存储蜕变关系（每主体一文件）。

存储结构：
    data/knowledge/mr_repository/{layer}/{subject_name}.yaml

每个文件格式：
    subject: torch.add
    layer: operator
    generated_at: "2026-01-01T00:00:00"
    mrs:
      - id: "abc-123"
        description: "..."
        transform_code: "lambda k: {**k, 'input': 2.0 * k['input']}"
        oracle_expr: "trans == 2.0 * orig"
        category: "linearity"
        tolerance: 1.0e-6
        layer: "operator"
        source: "llm"
        lifecycle_state: "proven"
        applicable_frameworks: ["pytorch"]
        checked: true
        proven: true
        verified: true
        analysis: "..."
        provenance:
          created_at: "2026-01-01T00:00:00+00:00"
          generator_id: "operator_llm_mr_generator"
          generator_version: "1.0"
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation

_VALID_LAYERS = {"operator", "model", "application"}


class MRRepository:
    """用户工作区 MR 仓库：每个主体（算子/模型/应用）的 MR 列表存为独立 YAML 文件。

    支持三层：operator / model / application。
    默认层为 operator，与旧版行为兼容。
    """

    def __init__(
        self,
        layer: str = "operator",
        repo_dir: str = "data/knowledge/mr_repository",
    ):
        if layer not in _VALID_LAYERS:
            raise ValueError(f"layer 必须是 {_VALID_LAYERS} 之一，得到: {layer!r}")
        self.layer = layer
        self._base_dir = Path(repo_dir)
        self.repo_dir = self._base_dir / layer
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📦 [REPO] MR repository [{layer}] at: {self.repo_dir}")

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _subject_file(self, subject_name: str) -> Path:
        safe_name = subject_name.replace("/", "__")
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
            "lifecycle_state": mr.lifecycle_state,
            "applicable_frameworks": mr.applicable_frameworks,
            "checked": mr.checked,
            "proven": mr.proven,
            "verified": mr.verified,
            "analysis": mr.analysis,
        }
        if mr.provenance:
            d["provenance"] = mr.provenance
        return d

    def _deserialize_mr(self, data: Dict) -> MetamorphicRelation:
        transform_code = data.get("transform_code", "")
        try:
            transform = eval(transform_code) if transform_code else lambda *args: args
        except Exception:
            transform = lambda *args: args

        mr = MetamorphicRelation(
            id=data["id"],
            description=data.get("description", ""),
            transform_code=transform_code,
            transform=transform,
            oracle_expr=data.get("oracle_expr", ""),
            category=data.get("category", "general"),
            tolerance=float(data.get("tolerance", 1e-6)),
            layer=data.get("layer", self.layer),
            source=data.get("source", ""),
            lifecycle_state=data.get("lifecycle_state", "pending"),
            applicable_frameworks=data.get("applicable_frameworks"),
            checked=data.get("checked"),
            proven=data.get("proven"),
            verified=data.get("verified", False),
            analysis=data.get("analysis", ""),
            provenance=data.get("provenance", {}),
        )
        return mr

    def _load_file(self, subject_name: str) -> Optional[Dict]:
        path = self._subject_file(subject_name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write_file(self, subject_name: str, data: Dict):
        path = self._subject_file(subject_name)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def save(
        self,
        subject_name: str,
        mrs: List[MetamorphicRelation],
        framework: Optional[str] = None,
    ) -> int:
        """将 MR 列表写入该主体的 YAML 文件（覆盖已有内容）。

        framework 为便利参数：若 MR 的 applicable_frameworks 为空则自动填入。
        """
        if not mrs:
            return 0

        for mr in mrs:
            if framework and not mr.applicable_frameworks:
                mr.applicable_frameworks = [framework]

        data = {
            "subject": subject_name,
            "layer": self.layer,
            "generated_at": datetime.now().isoformat(),
            "mrs": [self._serialize_mr(mr) for mr in mrs],
        }
        self._write_file(subject_name, data)
        logger.info(f"📦 [REPO] Saved {len(mrs)} MRs for [{self.layer}] {subject_name}")
        return len(mrs)

    def load(
        self,
        subject_name: str,
        framework: Optional[str] = None,
        min_quality: Optional[str] = None,
    ) -> List[MetamorphicRelation]:
        """从 YAML 文件加载 MR 列表。

        Args:
            subject_name:  主体名称（算子/模型/应用）
            framework:     可选框架过滤（applicable_frameworks 包含该值）
            min_quality:   可选最低质量过滤（"candidate"|"checked"|"proven"|"curated"）
        """
        data = self._load_file(subject_name)
        if not data:
            return []

        _quality_order = {"candidate": 1, "checked": 2, "proven": 3, "curated": 4, "retired": 0}
        min_score = _quality_order.get(min_quality or "candidate", 1)

        mrs = []
        for entry in data.get("mrs", []):
            try:
                mr = self._deserialize_mr(entry)
                if framework and mr.applicable_frameworks is not None:
                    if framework not in mr.applicable_frameworks:
                        continue
                if min_quality is not None:
                    ql = _quality_order.get(mr.quality_level, 1)
                    if ql < min_score:
                        continue
                mrs.append(mr)
            except Exception as e:
                logger.error(f"Error loading MR: {e}")

        logger.info(f"📦 [REPO] Loaded {len(mrs)} MRs for [{self.layer}] {subject_name}")
        return mrs

    def exists(self, subject_name: str) -> bool:
        return self._subject_file(subject_name).exists()

    def list_subjects(self) -> List[str]:
        """列出当前层所有主体名称（YAML 文件 stem）。"""
        return sorted(f.stem for f in self.repo_dir.glob("*.yaml"))

    # 向后兼容别名
    def list_operators(self) -> List[str]:
        return self.list_subjects()

    def delete(
        self,
        subject_name: str,
        mr_id: Optional[str] = None,
    ) -> int:
        """删除 MR 记录。mr_id=None 则删除整个主体文件，否则仅删除指定条目。"""
        path = self._subject_file(subject_name)
        if not path.exists():
            return 0

        if mr_id is None:
            data = self._load_file(subject_name)
            count = len(data.get("mrs", [])) if data else 0
            path.unlink()
            logger.info(f"📦 [REPO] Deleted all {count} MRs for [{self.layer}] '{subject_name}'")
            return count

        data = self._load_file(subject_name)
        if not data:
            return 0
        before = len(data.get("mrs", []))
        data["mrs"] = [m for m in data.get("mrs", []) if m.get("id") != mr_id]
        deleted = before - len(data["mrs"])
        if deleted > 0:
            self._write_file(subject_name, data)
        logger.info(
            f"📦 [REPO] Deleted {deleted} MR(s) for [{self.layer}] '{subject_name}' mr_id={mr_id}"
        )
        return deleted

    def retire(self, subject_name: str, mr_id: str) -> bool:
        """将指定 MR 标记为 retired（lifecycle_state=retired）。

        不删除记录，保留历史痕迹。

        Returns:
            True 如果找到并修改了该 MR，False 否则。
        """
        data = self._load_file(subject_name)
        if not data:
            return False
        found = False
        for entry in data.get("mrs", []):
            if entry.get("id") == mr_id:
                entry["lifecycle_state"] = "retired"
                entry["verified"] = False
                found = True
                break
        if found:
            self._write_file(subject_name, data)
            logger.info(
                f"📦 [REPO] Retired MR {mr_id} for [{self.layer}] '{subject_name}'"
            )
        return found

    def update_lifecycle(
        self, subject_name: str, mr_id: str, new_state: str
    ) -> bool:
        """更新指定 MR 的 lifecycle_state。

        Returns:
            True 如果成功更新，False 如果未找到。
        """
        _valid = {"pending", "checked", "proven", "curated", "retired"}
        if new_state not in _valid:
            raise ValueError(f"lifecycle_state 必须是 {_valid} 之一")
        data = self._load_file(subject_name)
        if not data:
            return False
        found = False
        for entry in data.get("mrs", []):
            if entry.get("id") == mr_id:
                entry["lifecycle_state"] = new_state
                if new_state == "curated":
                    entry["verified"] = True
                    entry["proven"] = True
                found = True
                break
        if found:
            self._write_file(subject_name, data)
        return found

    def list_subjects_by_framework(self, framework: str) -> List[str]:
        """列出包含指定框架 MR 的主体（applicable_frameworks 包含该框架，或为 None）。"""
        result = []
        for subject in self.list_subjects():
            for mr in self.load(subject):
                if mr.applicable_frameworks is None or framework in mr.applicable_frameworks:
                    result.append(subject)
                    break
        return sorted(result)

    # 向后兼容别名
    def list_operators_by_framework(self, framework: str) -> List[str]:
        return self.list_subjects_by_framework(framework)

    def get_mr_with_validation_status(
        self,
        subject_name: str,
        verified_only: bool = False,
        framework: Optional[str] = None,
    ) -> List[MetamorphicRelation]:
        """加载 MR，支持 verified_only 过滤。"""
        mrs = self.load(subject_name, framework=framework)
        if verified_only:
            mrs = [m for m in mrs if m.verified]
        return mrs

    def get_statistics(self, subject_name: Optional[str] = None) -> Dict:
        """统计 MR 数量信息，含质量等级分布。"""

        def _stats_for(mrs: List[MetamorphicRelation]) -> Dict:
            from collections import Counter
            quality_dist: Counter = Counter()
            for m in mrs:
                quality_dist[m.quality_level] += 1
            return {
                "total": len(mrs),
                "verified": sum(1 for m in mrs if m.verified),
                "unverified": sum(1 for m in mrs if not m.verified),
                "checked": sum(1 for m in mrs if m.checked is True),
                "proven": sum(1 for m in mrs if m.proven is True),
                "quality_dist": dict(quality_dist),
                "by_source": dict(Counter(m.source for m in mrs)),
                "retired": sum(1 for m in mrs if m.lifecycle_state == "retired"),
            }

        if subject_name:
            c = _stats_for(self.load(subject_name))
            return {
                "total_mrs": c["total"],
                "verified_mrs": c["verified"],
                "unverified_mrs": c["unverified"],
                "checked": c["checked"],
                "proven": c["proven"],
                "quality_dist": c["quality_dist"],
                "by_source": c["by_source"],
                "retired": c["retired"],
                "by_subject": {subject_name: c},
            }

        stats: Dict = {
            "total_mrs": 0,
            "verified_mrs": 0,
            "unverified_mrs": 0,
            "checked": 0,
            "proven": 0,
            "retired": 0,
            "quality_dist": {},
            "by_source": {},
            "by_subject": {},
        }
        from collections import Counter
        q_counter: Counter = Counter()
        s_counter: Counter = Counter()

        for subject in self.list_subjects():
            c = _stats_for(self.load(subject))
            stats["total_mrs"] += c["total"]
            stats["verified_mrs"] += c["verified"]
            stats["unverified_mrs"] += c["unverified"]
            stats["checked"] += c["checked"]
            stats["proven"] += c["proven"]
            stats["retired"] += c["retired"]
            q_counter.update(c["quality_dist"])
            s_counter.update(c["by_source"])
            stats["by_subject"][subject] = c
        stats["quality_dist"] = dict(q_counter)
        stats["by_source"] = dict(s_counter)
        return stats
