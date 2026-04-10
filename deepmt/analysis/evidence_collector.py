"""
缺陷证据包：捕获并持久化可复现的测试失败案例。

设计目的：
  - 让每一个检测到的失败都能被独立复现，而不仅是一次性日志记录
  - 生成「可直接粘贴运行」的 Python 复现脚本，是研究价值的核心体现
  - 为论文案例分析和答辩演示提供可信材料

数据流：
  BatchTestRunner（发现失败）
    → EvidencePack（捕获失败快照）
    → EvidenceCollector.save()（写入 data/evidence/<id>.json）
    → deepmt test evidence list/show（查看）

EvidencePack 包含：
  - 算子、MR、框架版本（来源溯源）
  - 输入张量的值（小张量）或摘要（大张量）
  - oracle 违例的具体数值
  - 可直接运行的 Python 复现脚本
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deepmt.core.logger import logger


# ── 框架版本获取 ──────────────────────────────────────────────────────────────

def _get_framework_version(framework: str) -> str:
    """获取框架版本字符串。"""
    if framework == "pytorch":
        try:
            import torch
            return torch.__version__
        except ImportError:
            return "unknown"
    return "unknown"


# ── 输入摘要工具 ──────────────────────────────────────────────────────────────

_MAX_ELEMENTS_FOR_CAPTURE = 200  # 超过此元素数时只记录摘要，不记录完整值


def _summarize_tensor(tensor: Any) -> Dict[str, Any]:
    """
    将框架张量转换为可序列化的摘要字典。

    - 小张量（≤ _MAX_ELEMENTS_FOR_CAPTURE 元素）：同时记录形状和完整值
    - 大张量：仅记录形状、dtype、min/max/mean
    """
    try:
        arr = np.asarray(tensor, dtype=float)
    except Exception:
        return {"error": "cannot convert to numpy"}

    shape = list(arr.shape)
    n_elements = int(np.prod(shape)) if shape else 1
    dtype_str = str(arr.dtype)

    summary: Dict[str, Any] = {
        "shape": shape,
        "dtype": dtype_str,
        "n_elements": n_elements,
    }

    if n_elements == 0:
        summary["values"] = []
        return summary

    try:
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            summary["min"] = float(finite.min())
            summary["max"] = float(finite.max())
            summary["mean"] = float(finite.mean())
    except Exception:
        pass

    if n_elements <= _MAX_ELEMENTS_FOR_CAPTURE:
        try:
            summary["values"] = arr.tolist()
        except Exception:
            pass

    return summary


# ── 复现脚本生成 ──────────────────────────────────────────────────────────────

def _generate_reproduce_script(
    operator_name: str,
    framework: str,
    framework_version: str,
    mr_description: str,
    transform_code: str,
    oracle_expr: str,
    input_summary: Dict[str, Any],
    actual_diff: float,
    tolerance: float,
) -> str:
    """
    生成可直接运行的 Python 复现脚本。

    脚本包含：
      - 必要的 import
      - 输入张量构造（使用捕获到的值，或生成相同形状的随机张量）
      - 原始算子调用
      - MR 变换与变换后算子调用
      - Oracle 验证与差值打印
    """
    lines = [
        '"""',
        f"DeepMT 失败复现脚本",
        f"算子: {operator_name}",
        f"MR:  {mr_description}",
        f"框架: {framework} {framework_version}",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        '"""',
        "",
    ]

    # imports
    if framework == "pytorch":
        lines += ["import torch", "import torch.nn.functional as F", ""]
    else:
        lines += [f"# import {framework} here", ""]

    # 输入构造
    shape = input_summary.get("shape", [4, 4])
    dtype_str = input_summary.get("dtype", "float64")
    values = input_summary.get("values")

    lines.append("# ── 输入数据 ──────────────────────────────────────────")
    if values is not None and framework == "pytorch":
        lines.append(f"input_data = torch.tensor({values}, dtype=torch.float32)")
    elif framework == "pytorch":
        # 无法还原精确值，使用随机张量
        lines.append(f"# 原始失败输入 shape={shape}，此处使用随机张量近似")
        lines.append(f"torch.manual_seed(42)")
        lines.append(f"input_data = torch.randn({shape})")
    lines.append("")

    # 算子调用
    lines.append("# ── 原始算子调用 ────────────────────────────────────────")
    lines.append("kwargs = {'input': input_data}")
    lines.append(f"orig = {operator_name}(**kwargs)")
    lines.append("")

    # MR 变换
    lines.append("# ── MR 变换 ─────────────────────────────────────────────")
    lines.append(f"# 变换代码: {transform_code}")
    lines.append(f"transform = {transform_code}")
    lines.append("kwargs_trans = transform(kwargs)")
    lines.append(f"trans = {operator_name}(**kwargs_trans)")
    lines.append("")

    # Oracle 验证
    lines.append("# ── Oracle 验证 ──────────────────────────────────────────")
    lines.append(f"x = input_data  # oracle_expr 中 x 变量")
    lines.append(f"oracle_expr = {oracle_expr!r}")
    lines.append(f"tolerance   = {tolerance}")
    lines.append(f"# 实测最大绝对差: {actual_diff:.6g}  (阈值: {tolerance})")
    lines.append("")
    if framework == "pytorch":
        lines.append("print(f'orig:  {orig}')")
        lines.append("print(f'trans: {trans}')")
        lines.append("print(f'oracle: {oracle_expr!r}')")
        lines.append("# 手动验证：")
        lines.append("# eval(oracle_expr) 时 orig/trans/x 均为 torch.Tensor")

    return "\n".join(lines)


# ── EvidencePack ──────────────────────────────────────────────────────────────


@dataclass
class EvidencePack:
    """
    单次测试失败的可复现证据包。

    字段分三组：
      溯源信息   — 定位问题来源（算子、MR、框架、时间）
      数值证据   — 说明失败的具体数值（diff、oracle、输入摘要）
      复现材料   — 快速复现所需的脚本和命令
    """

    # ── 唯一标识 ──────────────────────────────────────────────────────────────
    evidence_id: str
    timestamp: str

    # ── 溯源信息 ──────────────────────────────────────────────────────────────
    operator: str
    framework: str
    framework_version: str
    mr_id: str
    mr_description: str
    transform_code: str
    oracle_expr: str

    # ── 数值证据 ──────────────────────────────────────────────────────────────
    input_summary: Dict[str, Any]    # shape/dtype/min/max/mean，小张量含 values
    actual_diff: float
    tolerance: float
    detail: str                       # 失败原因（NUMERICAL_DEVIATION / SHAPE_MISMATCH 等）

    # ── 复现材料 ──────────────────────────────────────────────────────────────
    reproduce_script: str             # 可直接运行的 Python 脚本

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidencePack":
        return cls(**d)


# ── EvidenceCollector ─────────────────────────────────────────────────────────


class EvidenceCollector:
    """
    证据包的保存与查询器。

    每个证据包存为独立的 JSON 文件：
        data/evidence/<evidence_id>.json

    用法示例：
        collector = EvidenceCollector()
        pack = collector.create(
            operator="torch.nn.functional.relu",
            framework="pytorch",
            mr_id="...",
            mr_description="...",
            transform_code="lambda k: {**k, 'input': -k['input']}",
            oracle_expr="orig + trans == abs(x)",
            input_tensor=x_tensor,
            actual_diff=0.5,
            tolerance=1e-6,
            detail="NUMERICAL_DEVIATION: max_abs=0.5",
        )
        collector.save(pack)
    """

    DEFAULT_DIR = Path("data/evidence")

    def __init__(self, evidence_dir: Optional[str] = None):
        self.evidence_dir = Path(evidence_dir) if evidence_dir else self.DEFAULT_DIR
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    # ── 创建与保存 ─────────────────────────────────────────────────────────────

    def create(
        self,
        operator: str,
        framework: str,
        mr_id: str,
        mr_description: str,
        transform_code: str,
        oracle_expr: str,
        input_tensor: Any,
        actual_diff: float,
        tolerance: float,
        detail: str,
    ) -> EvidencePack:
        """
        根据一次测试失败的现场数据，创建 EvidencePack（不自动保存）。

        Args:
            input_tensor: 触发失败的输入张量（主输入，即 kwargs['input']）
        """
        evidence_id = str(uuid.uuid4())[:12]  # 短 ID，便于引用
        timestamp = datetime.now().isoformat()
        fw_version = _get_framework_version(framework)

        input_summary = _summarize_tensor(input_tensor)

        reproduce_script = _generate_reproduce_script(
            operator_name=operator,
            framework=framework,
            framework_version=fw_version,
            mr_description=mr_description,
            transform_code=transform_code,
            oracle_expr=oracle_expr,
            input_summary=input_summary,
            actual_diff=actual_diff,
            tolerance=tolerance,
        )

        return EvidencePack(
            evidence_id=evidence_id,
            timestamp=timestamp,
            operator=operator,
            framework=framework,
            framework_version=fw_version,
            mr_id=mr_id,
            mr_description=mr_description,
            transform_code=transform_code,
            oracle_expr=oracle_expr,
            input_summary=input_summary,
            actual_diff=actual_diff,
            tolerance=tolerance,
            detail=detail,
            reproduce_script=reproduce_script,
        )

    def save(self, pack: EvidencePack) -> Path:
        """将证据包写入 JSON 文件，返回文件路径。"""
        path = self.evidence_dir / f"{pack.evidence_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pack.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[EVIDENCE] Saved: {path}")
        return path

    # ── 查询 ───────────────────────────────────────────────────────────────────

    def list_all(
        self,
        operator: Optional[str] = None,
        framework: Optional[str] = None,
        limit: int = 0,
    ) -> List[EvidencePack]:
        """
        列出所有（或过滤后的）证据包，按时间倒序排列。

        Args:
            operator:  按算子名称过滤；None 表示全部
            framework: 按框架过滤；None 表示全部
            limit:     最多返回条数（0 = 不限）

        Returns:
            EvidencePack 列表，按 timestamp 倒序
        """
        packs = []
        for path in self.evidence_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                pack = EvidencePack.from_dict(d)
                if operator and pack.operator != operator:
                    continue
                if framework and pack.framework != framework:
                    continue
                packs.append(pack)
            except Exception as e:
                logger.warning(f"[EVIDENCE] Cannot load {path}: {e}")

        packs.sort(key=lambda p: p.timestamp, reverse=True)
        if limit > 0:
            packs = packs[:limit]
        return packs

    def load(self, evidence_id: str) -> Optional[EvidencePack]:
        """加载指定 ID 的证据包；未找到时返回 None。"""
        path = self.evidence_dir / f"{evidence_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return EvidencePack.from_dict(json.load(f))
        except Exception as e:
            logger.error(f"[EVIDENCE] Cannot load {path}: {e}")
            return None

    def count(self) -> int:
        """返回证据包总数。"""
        return len(list(self.evidence_dir.glob("*.json")))
