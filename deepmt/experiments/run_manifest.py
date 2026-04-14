"""
实验运行清单（Run Manifest）。

职责：
  记录单次实验运行的完整元数据，确保实验可重跑、可追溯、可对比。

一个 RunManifest 对应一次完整实验运行，包含：
  - 实验 ID 与时间戳
  - 随机种子（全局固定，保证可复现）
  - 目标 RQ 集合
  - benchmark 范围（算子/模型/应用）
  - 环境快照（EnvironmentSnapshot）
  - 运行状态（pending / running / completed / failed）
  - 结果摘要路径

存储格式：JSON（存入 data/experiments/runs/<run_id>.json）

用法::

    from deepmt.experiments.run_manifest import RunManifest, RunManifestManager

    # 创建
    mgr = RunManifestManager()
    manifest = mgr.create(rqs=["rq1","rq2"], seed=42)
    mgr.save(manifest)

    # 加载
    m = mgr.load(manifest.run_id)
    print(m.run_id, m.seed, m.status)

    # 列出所有
    for m in mgr.list_all():
        print(m.run_id, m.created_at, m.status)
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepmt.core.logger import logger

_DEFAULT_RUNS_DIR = Path("data/experiments/runs")

# ── 全局推荐种子 ──────────────────────────────────────────────────────────────
#
# 论文实验统一使用此种子，确保结果可复现。
# 若需要多次独立重跑，可传入不同 seed 并在 manifest 中记录。
GLOBAL_SEED = 42


@dataclass
class RunManifest:
    """单次实验运行的完整元数据。"""

    run_id: str
    """唯一运行 ID（UUID4 短格式）。"""

    created_at: str
    """创建时间（ISO 格式）。"""

    seed: int
    """随机种子。"""

    rqs: List[str]
    """本次运行覆盖的 RQ 集合（如 ['rq1','rq2']）。"""

    benchmark_operators: List[str] = field(default_factory=list)
    """本次运行的算子 benchmark 名称列表。"""

    benchmark_models: List[str] = field(default_factory=list)
    """本次运行的模型 benchmark 名称列表。"""

    benchmark_applications: List[str] = field(default_factory=list)
    """本次运行的应用 benchmark 名称列表。"""

    status: str = "pending"
    """运行状态：pending / running / completed / failed。"""

    started_at: Optional[str] = None
    """实际开始时间。"""

    completed_at: Optional[str] = None
    """完成时间。"""

    environment: Optional[Dict[str, Any]] = None
    """环境快照（EnvironmentSnapshot.to_dict()）。"""

    result_summary_path: Optional[str] = None
    """结果摘要 JSON 文件路径（相对于项目根）。"""

    notes: str = ""
    """附加备注。"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "rqs": self.rqs,
            "benchmark_operators": self.benchmark_operators,
            "benchmark_models": self.benchmark_models,
            "benchmark_applications": self.benchmark_applications,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "environment": self.environment,
            "result_summary_path": self.result_summary_path,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunManifest":
        return cls(
            run_id=d["run_id"],
            created_at=d["created_at"],
            seed=d.get("seed", GLOBAL_SEED),
            rqs=d.get("rqs", []),
            benchmark_operators=d.get("benchmark_operators", []),
            benchmark_models=d.get("benchmark_models", []),
            benchmark_applications=d.get("benchmark_applications", []),
            status=d.get("status", "pending"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            environment=d.get("environment"),
            result_summary_path=d.get("result_summary_path"),
            notes=d.get("notes", ""),
        )

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = datetime.now().isoformat()

    def mark_completed(self, result_path: Optional[str] = None) -> None:
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        if result_path is not None:
            self.result_summary_path = result_path

    def mark_failed(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()

    def format_text(self) -> str:
        lines = [
            f"Run Manifest [{self.run_id}]",
            f"  Created:     {self.created_at[:19]}",
            f"  Seed:        {self.seed}",
            f"  Status:      {self.status}",
            f"  RQs:         {', '.join(self.rqs) or '(全部)'}",
            f"  Operators:   {len(self.benchmark_operators)} 个",
            f"  Models:      {len(self.benchmark_models)} 个",
            f"  Applications:{len(self.benchmark_applications)} 个",
        ]
        if self.result_summary_path:
            lines.append(f"  Results:     {self.result_summary_path}")
        if self.notes:
            lines.append(f"  Notes:       {self.notes}")
        return "\n".join(lines)


class RunManifestManager:
    """
    RunManifest 的持久化管理器。

    负责 manifest 的创建、保存、加载和列举。
    存储目录：data/experiments/runs/（自动创建）。
    """

    def __init__(self, runs_dir: Optional[Path] = None):
        self._dir = Path(runs_dir) if runs_dir else _DEFAULT_RUNS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        rqs: Optional[List[str]] = None,
        seed: int = GLOBAL_SEED,
        capture_env: bool = True,
        notes: str = "",
    ) -> RunManifest:
        """
        创建新的 RunManifest。

        Args:
            rqs:         目标 RQ 列表（默认全部 ['rq1','rq2','rq3','rq4']）
            seed:        随机种子
            capture_env: 是否立即捕获环境快照
            notes:       附加备注
        """
        from deepmt.benchmarks.suite import BenchmarkSuite

        if rqs is None:
            rqs = ["rq1", "rq2", "rq3", "rq4"]

        suite = BenchmarkSuite()
        run_id = uuid.uuid4().hex[:12]

        env_dict = None
        if capture_env:
            try:
                from deepmt.experiments.environment_recorder import EnvironmentRecorder
                env_dict = EnvironmentRecorder().capture().to_dict()
            except Exception as e:
                logger.warning(f"[RunManifest] 环境捕获失败: {e}")

        return RunManifest(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            seed=seed,
            rqs=rqs,
            benchmark_operators=suite.operator_names(),
            benchmark_models=suite.model_names(),
            benchmark_applications=suite.application_names(),
            environment=env_dict,
            notes=notes,
        )

    def save(self, manifest: RunManifest) -> Path:
        """将 manifest 序列化为 JSON 并写入磁盘，返回文件路径。"""
        path = self._dir / f"{manifest.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug(f"[RunManifest] 已保存: {path}")
        return path

    def load(self, run_id: str) -> Optional[RunManifest]:
        """按 run_id 加载 manifest，未找到返回 None。"""
        path = self._dir / f"{run_id}.json"
        if not path.exists():
            logger.warning(f"[RunManifest] 未找到: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return RunManifest.from_dict(json.load(f))

    def list_all(self) -> List[RunManifest]:
        """加载并返回所有 manifest（按创建时间升序）。"""
        manifests = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    manifests.append(RunManifest.from_dict(json.load(f)))
            except Exception as e:
                logger.warning(f"[RunManifest] 读取 {p.name} 失败: {e}")
        manifests.sort(key=lambda m: m.created_at)
        return manifests

    def update(self, manifest: RunManifest) -> None:
        """更新（覆盖写入）已有 manifest。"""
        self.save(manifest)
