"""MR 项目库：以单一 YAML 文件存储经过验证的 MR，由 git 追踪。

存储结构：
    data/knowledge/mr_library/operator/operator.yaml

文件格式（以算子名为键的 mapping）：
    torch.add:
      - id: "abc-123"
        description: "Addition is commutative"
        transform_code: "lambda k: {**k, 'input': k['other'], 'other': k['input']}"
        oracle_expr: "orig == trans"
        applicable_frameworks: ["pytorch"]   # null 表示通用
        # tolerance 仅在非默认值（1e-6）时写入

只存储 verified=True 的 MR，不含 analysis、timestamps 等噪音字段。
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import yaml

from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation

if TYPE_CHECKING:
    from deepmt.mr_generator.base.mr_repository import MRRepository

_DEFAULT_TOLERANCE = 1e-6


class MRLibrary:
    """项目级 MR 库：只存储 verified=True 的 MR，YAML 格式干净，由 git 追踪。"""

    def __init__(self, layer: str = "operator", library_dir: str = "data/knowledge/mr_library"):
        self.layer = layer
        self.library_dir = Path(library_dir)
        self._yaml_path = self.library_dir / layer / f"{layer}.yaml"
        self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📚 [LIB] MR library at: {self._yaml_path}")

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _load_raw(self) -> Dict[str, List[Dict]]:
        if not self._yaml_path.exists():
            return {}
        with open(self._yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _write_raw(self, data: Dict[str, List[Dict]]):
        with open(self._yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=True, default_flow_style=False)

    def _serialize_mr(self, mr: MetamorphicRelation) -> Dict:
        d: Dict = {
            "id": mr.id,
            "description": mr.description,
            "transform_code": mr.transform_code,
            "oracle_expr": mr.oracle_expr,
        }
        if mr.applicable_frameworks is not None:
            d["applicable_frameworks"] = mr.applicable_frameworks
        if abs(mr.tolerance - _DEFAULT_TOLERANCE) > 1e-12:
            d["tolerance"] = mr.tolerance
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
            tolerance=float(data.get("tolerance", _DEFAULT_TOLERANCE)),
            applicable_frameworks=data.get("applicable_frameworks"),
            verified=True,
            checked=True,
            proven=True,
        )

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def save(self, operator_name: str, mrs: List[MetamorphicRelation]) -> int:
        """将 verified=True 的 MR 写入项目库（覆盖该算子的现有条目）。"""
        verified = [m for m in mrs if m.verified]
        if not verified:
            logger.warning(f"📚 [LIB] No verified MRs to save for '{operator_name}'")
            return 0
        data = self._load_raw()
        data[operator_name] = [self._serialize_mr(m) for m in verified]
        self._write_raw(data)
        logger.info(f"📚 [LIB] Saved {len(verified)} MRs for operator: {operator_name}")
        return len(verified)

    def load(self, operator_name: Optional[str] = None) -> List[MetamorphicRelation]:
        """加载 MR，operator_name=None 时加载全部。"""
        data = self._load_raw()
        result = []
        items = [(operator_name, data.get(operator_name, []))] if operator_name else data.items()
        for _op, entries in items:
            for entry in entries:
                try:
                    result.append(self._deserialize_mr(entry))
                except Exception as e:
                    logger.error(f"📚 [LIB] Error loading MR: {e}")
        return result

    def exists(self, operator_name: str) -> bool:
        return operator_name in self._load_raw()

    def list_operators(self) -> List[str]:
        return sorted(self._load_raw().keys())

    def promote_from_repository(self, operator_name: str, repo: "MRRepository") -> int:
        """从用户仓库将 verified=True 的 MR 迁移复制到项目库（剥离噪音字段）。"""
        mrs = repo.load(operator_name)
        verified = [m for m in mrs if m.verified]
        if not verified:
            logger.warning(f"📚 [LIB] No verified MRs found in repo for '{operator_name}'")
            return 0
        count = self.save(operator_name, verified)
        logger.info(f"📚 [LIB] Promoted {count} MRs from repo to library for '{operator_name}'")
        return count
