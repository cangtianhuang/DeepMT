"""MR 来源溯源信息（K3）。

ProvenanceInfo 记录一条 MR 的生成元数据：
    created_at      — 生成时间（ISO8601 字符串）
    generator_id    — 生成器标识（如 "operator_llm_mr_generator"）
    generator_version — 生成器版本（如 "1.0"）
    source_detail   — 来源细节（如 LLM 模型名、模板 ID、规则版本）
    prompt_hash     — prompt 内容的哈希（用于 LLM 生成的可重现性追踪）
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ProvenanceInfo:
    """MR 来源信息。

    可直接转为 dict 后写入 MetamorphicRelation.provenance 字段。
    """

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    generator_id: str = ""
    generator_version: str = "1.0"
    source_detail: str = ""
    prompt_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（None 值被省略）。"""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != ""}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceInfo":
        """从 dict 反序列化。"""
        return cls(
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            generator_id=data.get("generator_id", ""),
            generator_version=data.get("generator_version", "1.0"),
            source_detail=data.get("source_detail", ""),
            prompt_hash=data.get("prompt_hash"),
        )


def build_provenance(
    generator_id: str,
    generator_version: str = "1.0",
    source_detail: str = "",
    prompt_text: Optional[str] = None,
) -> Dict[str, Any]:
    """构建 provenance dict，可直接赋给 MetamorphicRelation.provenance。

    Args:
        generator_id:       生成器标识（如 "model_mr_generator"）
        generator_version:  生成器版本
        source_detail:      来源细节（如 LLM 模型名、模板 ID）
        prompt_text:        可选的 prompt 原文，用于计算 sha256 哈希

    Returns:
        符合 ProvenanceInfo 字段的 dict
    """
    info = ProvenanceInfo(
        generator_id=generator_id,
        generator_version=generator_version,
        source_detail=source_detail,
        prompt_hash=_sha256_short(prompt_text) if prompt_text else None,
    )
    return info.to_dict()


def _sha256_short(text: str, length: int = 8) -> str:
    """计算文本的 SHA-256 前 length 位十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
