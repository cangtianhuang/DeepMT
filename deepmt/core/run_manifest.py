"""运行清单：追踪单次测试运行的完整上下文。

每次执行 `deepmt test batch` 或任何测试命令时，应创建一个 RunManifest 实例，
绑定到本次运行的所有 test_results 行上（通过 run_id 关联），从而支持：
  - 实验回放（已知 seed + framework_version + env）
  - 跨版本比较（同 MR，不同 framework_version）
  - 论文数据聚合（run 级别的统计与索引）
"""

import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


def _default_env() -> Dict[str, str]:
    """采集当前运行环境的最小摘要，用于可复现性记录。"""
    return {
        "python": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
    }


@dataclass
class RunManifest:
    """单次测试运行的完整上下文记录。

    Attributes:
        run_id:            全局唯一运行标识（UUID4）
        subject_name:      被测主体名称（如 "torch.add"）
        subject_type:      被测主体层次（"operator" / "model" / "application"）
        framework:         测试框架名称（如 "pytorch"）
        framework_version: 框架版本字符串（如 "2.3.0"）；空字符串表示未知
        random_seed:       随机输入生成所用种子；None 表示未固定
        n_samples:         本次运行的输入样本数
        mr_ids:            本次运行覆盖的 MR ID 列表
        env_summary:       运行环境摘要（python 版本、OS、机器类型等）
        timestamp:         运行开始时间（ISO 8601 字符串）
        notes:             可选备注（如实验名称、RQ 标签）
    """

    subject_name: str
    subject_type: str = "operator"
    framework: str = "pytorch"
    framework_version: str = ""
    random_seed: Optional[int] = None
    n_samples: int = 0
    mr_ids: List[str] = field(default_factory=list)
    env_summary: Dict[str, str] = field(default_factory=_default_env)
    notes: str = ""

    # 自动填充字段
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """序列化为字典，用于 JSON 导出或 SQLite 存储。"""
        return {
            "run_id": self.run_id,
            "subject_name": self.subject_name,
            "subject_type": self.subject_type,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "random_seed": self.random_seed,
            "n_samples": self.n_samples,
            "mr_ids": self.mr_ids,
            "env_summary": self.env_summary,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RunManifest":
        """从字典反序列化，用于从 SQLite 或 JSON 恢复。"""
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            subject_name=data["subject_name"],
            subject_type=data.get("subject_type", "operator"),
            framework=data.get("framework", "pytorch"),
            framework_version=data.get("framework_version", ""),
            random_seed=data.get("random_seed"),
            n_samples=data.get("n_samples", 0),
            mr_ids=data.get("mr_ids", []),
            env_summary=data.get("env_summary", _default_env()),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
