"""
跨框架一致性测试器：验证不同框架在等价算子上的行为一致性。

设计目的：
  - 为论文 RQ3 提供跨框架一致性实验数据
  - 识别"合理的数值差异"和"可疑的行为不一致"
  - 验证同一条 MR 在不同框架中是否得出相同结论

核心概念：
  - 一致性（consistency）：两框架对同一输入执行同一 MR，结论相同（both pass 或 both fail）
  - 输出差异（output_diff）：两框架在相同输入上的实际数值差距，反映实现细节差异

数据流：
  CrossFrameworkTester.compare_operator(op, f1, f2)
    → CrossConsistencyResult（每条 MR 一个）
    → CrossSessionResult（一次对比实验的完整记录）
    → 可选：save() 持久化到 data/cross_results/<session_id>.json

框架支持：
  - "pytorch"：PyTorchPlugin（主链）
  - "numpy"：NumpyPlugin（数值参考后端，无需额外安装）
  - 其他已注册框架：通过 PluginsManager.get_backend() 加载
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deepmt.analysis.mr_prechecker import MRPreChecker
from deepmt.analysis.mr_verifier import MRVerifier
from deepmt.analysis.random_generator import RandomGenerator
from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
from deepmt.plugins.framework_plugin import FrameworkPlugin


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class CrossConsistencyResult:
    """
    单条 MR 在两个框架之间的一致性对比结果。

    字段分三组：
      标识信息   — 算子、MR、两框架名称
      一致性统计 — 两框架结论是否一致（both_pass / only_f1_pass / only_f2_pass / both_fail）
      输出差异   — 相同输入下两框架实际数值差距
    """

    # ── 标识信息 ──────────────────────────────────────────────────────────────
    operator: str
    framework1: str
    framework2: str
    mr_id: str
    mr_description: str
    oracle_expr: str
    n_samples: int

    # ── 一致性统计 ────────────────────────────────────────────────────────────
    both_pass: int        # 两框架均满足 MR
    only_f1_pass: int     # 仅 f1 满足（f2 不一致）
    only_f2_pass: int     # 仅 f2 满足（f1 不一致）
    both_fail: int        # 两框架均不满足（MR 对两者均不成立）
    errors: int           # 执行异常数

    # ── 输出差异 ──────────────────────────────────────────────────────────────
    output_max_diff: float          # max|f1_output - f2_output| 的最大值（跨所有样本）
    output_mean_diff: float         # 各样本 max 差值的均值
    output_close: bool              # 全部样本输出差 < output_close_threshold

    @property
    def total_valid(self) -> int:
        return self.both_pass + self.only_f1_pass + self.only_f2_pass + self.both_fail

    @property
    def consistency_rate(self) -> float:
        """两框架结论一致的比例（both_pass + both_fail）。"""
        return (self.both_pass + self.both_fail) / self.total_valid if self.total_valid > 0 else 0.0

    @property
    def f1_pass_rate(self) -> float:
        return (self.both_pass + self.only_f1_pass) / self.total_valid if self.total_valid > 0 else 0.0

    @property
    def f2_pass_rate(self) -> float:
        return (self.both_pass + self.only_f2_pass) / self.total_valid if self.total_valid > 0 else 0.0

    @property
    def inconsistent_cases(self) -> int:
        """结论不一致（仅一个框架通过）的样本数。"""
        return self.only_f1_pass + self.only_f2_pass

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["consistency_rate"] = round(self.consistency_rate, 4)
        d["f1_pass_rate"]     = round(self.f1_pass_rate, 4)
        d["f2_pass_rate"]     = round(self.f2_pass_rate, 4)
        d["inconsistent_cases"] = self.inconsistent_cases
        return d


@dataclass
class CrossSessionResult:
    """
    单次跨框架一致性实验的完整记录（一个算子 × 两个框架）。

    可持久化为 JSON，用于 ExperimentOrganizer 汇总 RQ3 数据。
    """

    session_id: str
    timestamp: str
    operator: str
    framework1: str
    framework2: str
    n_samples: int
    mr_results: List[CrossConsistencyResult] = field(default_factory=list)

    @property
    def mr_count(self) -> int:
        return len(self.mr_results)

    @property
    def overall_consistency_rate(self) -> float:
        if not self.mr_results:
            return 0.0
        return sum(r.consistency_rate for r in self.mr_results) / len(self.mr_results)

    @property
    def output_max_diff(self) -> float:
        diffs = [r.output_max_diff for r in self.mr_results if r.output_max_diff == r.output_max_diff]
        return max(diffs) if diffs else float("nan")

    @property
    def inconsistent_mr_count(self) -> int:
        """有至少一个不一致样本的 MR 数量。"""
        return sum(1 for r in self.mr_results if r.inconsistent_cases > 0)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "operator": self.operator,
            "framework1": self.framework1,
            "framework2": self.framework2,
            "n_samples": self.n_samples,
            "mr_count": self.mr_count,
            "overall_consistency_rate": round(self.overall_consistency_rate, 4),
            "output_max_diff": self.output_max_diff,
            "inconsistent_mr_count": self.inconsistent_mr_count,
            "mr_results": [r.to_dict() for r in self.mr_results],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CrossSessionResult":
        mr_results = [
            CrossConsistencyResult(**{
                k: v for k, v in r.items()
                if k in CrossConsistencyResult.__dataclass_fields__
            })
            for r in d.get("mr_results", [])
        ]
        return cls(
            session_id=d["session_id"],
            timestamp=d["timestamp"],
            operator=d["operator"],
            framework1=d["framework1"],
            framework2=d["framework2"],
            n_samples=d["n_samples"],
            mr_results=mr_results,
        )


# ── CrossFrameworkTester ──────────────────────────────────────────────────────


class CrossFrameworkTester:
    """
    跨框架一致性测试器。

    对同一算子的所有 MR，在两个不同框架后端上分别执行，
    统计：
      1. 两框架对 MR 的结论一致性（both_pass / inconsistent / both_fail）
      2. 两框架在相同输入上的数值输出差异

    用法示例：
        tester = CrossFrameworkTester()
        session = tester.compare_operator(
            "torch.nn.functional.relu",
            framework1="pytorch",
            framework2="numpy",
            n_samples=20,
        )
        print(f"一致率: {session.overall_consistency_rate:.1%}")
        tester.save(session)

    CLI 等价：
        deepmt test cross torch.nn.functional.relu
        deepmt test cross torch.exp --framework1 pytorch --framework2 numpy --n-samples 30
    """

    DEFAULT_RESULTS_DIR = Path("data/cross_results")
    OUTPUT_CLOSE_THRESHOLD = 1e-3  # 输出差 < 此阈值视为"数值接近"

    def __init__(
        self,
        repo: Optional[MRRepository] = None,
        catalog: Optional[OperatorCatalog] = None,
        results_dir: Optional[str] = None,
    ):
        self.repo = repo or MRRepository()
        self.catalog = catalog or OperatorCatalog()
        self.random_gen = RandomGenerator()
        self.verifier = MRVerifier()
        self.results_dir = Path(results_dir) if results_dir else self.DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def compare_operator(
        self,
        operator_name: str,
        framework1: str = "pytorch",
        framework2: str = "numpy",
        n_samples: int = 20,
        verified_only: bool = False,
    ) -> CrossSessionResult:
        """
        对单个算子执行跨框架一致性测试。

        Args:
            operator_name: 算子名称（与 MR 知识库中的键一致）
            framework1:    第一框架（默认 pytorch）
            framework2:    第二框架（默认 numpy）
            n_samples:     每条 MR 的测试样本数
            verified_only: 仅使用已验证的 MR

        Returns:
            CrossSessionResult，含每条 MR 的对比结果
        """
        logger.info(f"[CROSS] {operator_name} | {framework1} vs {framework2} | n_samples={n_samples}")

        backend1 = self._get_backend(framework1)
        backend2 = self._get_backend(framework2)

        mrs = self.repo.load(operator_name)
        if verified_only:
            mrs = [m for m in mrs if m.verified]

        if not mrs:
            logger.warning(f"[CROSS] No MRs found for {operator_name!r}")
            return CrossSessionResult(
                session_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                operator=operator_name,
                framework1=framework1,
                framework2=framework2,
                n_samples=n_samples,
                mr_results=[],
            )

        # 获取 input_specs（用于输入生成）
        entry = self.catalog.get_operator_info(framework1, operator_name)
        input_specs = entry.input_specs if (entry and entry.input_specs) else []

        mr_results = []
        for mr in mrs:
            result = self._compare_single_mr(
                mr=mr,
                operator_name=operator_name,
                backend1=backend1,
                backend2=backend2,
                framework1=framework1,
                framework2=framework2,
                input_specs=input_specs,
                n_samples=n_samples,
            )
            mr_results.append(result)

        session = CrossSessionResult(
            session_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            operator=operator_name,
            framework1=framework1,
            framework2=framework2,
            n_samples=n_samples,
            mr_results=mr_results,
        )
        logger.info(
            f"[CROSS] {operator_name}: consistency={session.overall_consistency_rate:.1%}"
            f"  mr_count={session.mr_count}"
            f"  output_max_diff={session.output_max_diff:.4g}"
        )
        return session

    def compare_batch(
        self,
        operators: Optional[List[str]] = None,
        framework1: str = "pytorch",
        framework2: str = "numpy",
        n_samples: int = 20,
        verified_only: bool = False,
    ) -> List[CrossSessionResult]:
        """
        批量跨框架一致性测试。

        Args:
            operators: 算子名称列表；None 时自动从 MR 知识库读取
        """
        if operators is None:
            operators = self.repo.list_operators_by_framework(framework1)

        # 过滤掉 framework2 不支持的算子
        backend2 = self._get_backend(framework2)
        supported = []
        for op in operators:
            try:
                backend2._resolve_operator(op)
                supported.append(op)
            except (ValueError, KeyError):
                logger.debug(f"[CROSS] {framework2} does not support {op!r}, skipping")

        results = []
        for op in supported:
            session = self.compare_operator(
                op, framework1, framework2, n_samples=n_samples, verified_only=verified_only
            )
            results.append(session)
        return results

    def save(self, session: CrossSessionResult) -> Path:
        """将会话结果持久化为 JSON 文件，返回路径。"""
        path = self.results_dir / f"{session.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[CROSS] Saved session: {path}")
        return path

    def load_all(self) -> List[CrossSessionResult]:
        """加载 results_dir 中全部会话结果，按时间倒序排列。"""
        sessions = []
        for path in self.results_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    sessions.append(CrossSessionResult.from_dict(json.load(f)))
            except Exception as e:
                logger.warning(f"[CROSS] Cannot load {path}: {e}")
        sessions.sort(key=lambda s: s.timestamp, reverse=True)
        return sessions

    def format_text(self, sessions: List[CrossSessionResult]) -> str:
        """格式化为终端可读文本。"""
        if not sessions:
            return "无跨框架一致性实验记录。运行 deepmt test cross <operator> 生成。"

        lines = [f"\n跨框架一致性报告（共 {len(sessions)} 条实验）", "─" * 72]
        for s in sessions:
            lines.append(
                f"\n  {s.operator}  [{s.framework1} vs {s.framework2}]"
                f"  时间: {s.timestamp[:16]}"
            )
            lines.append(
                f"    整体一致率: {s.overall_consistency_rate:.1%}"
                f"  MR 数: {s.mr_count}"
                f"  输出最大差: {s.output_max_diff:.4g}"
            )
            for r in s.mr_results:
                mark = "≈" if r.consistency_rate >= 0.9 else ("!" if r.inconsistent_cases > 0 else "≠")
                lines.append(
                    f"    {mark} {r.mr_description[:52]}"
                    f"  consistency={r.consistency_rate:.0%}"
                    f"  diff={r.output_max_diff:.3g}"
                )
        lines.append("\n" + "─" * 72)
        lines.append(
            f"  ≈ 高度一致（≥90%）  ! 存在不一致  ≠ 低一致率"
        )
        return "\n".join(lines)

    # ── 私有实现 ──────────────────────────────────────────────────────────────

    def _get_backend(self, framework: str) -> FrameworkPlugin:
        """获取框架后端实例，支持 'numpy' 特殊框架。"""
        if framework.lower() == "numpy":
            from deepmt.plugins.numpy_plugin import NumpyPlugin
            return NumpyPlugin()
        from deepmt.core.plugins_manager import get_plugins_manager
        return get_plugins_manager().get_backend(framework)

    def _compare_single_mr(
        self,
        mr: MetamorphicRelation,
        operator_name: str,
        backend1: FrameworkPlugin,
        backend2: FrameworkPlugin,
        framework1: str,
        framework2: str,
        input_specs: List[Dict],
        n_samples: int,
    ) -> CrossConsistencyResult:
        """对单条 MR 执行跨框架对比，返回 CrossConsistencyResult。"""

        # 解析两个框架的算子函数
        try:
            op_f1 = backend1._resolve_operator(operator_name)
        except (ValueError, KeyError) as e:
            logger.warning(f"[CROSS] {framework1} cannot resolve {operator_name!r}: {e}")
            return self._empty_result(operator_name, framework1, framework2, mr, n_samples, "unsupported_f1")

        try:
            op_f2 = backend2._resolve_operator(operator_name)
        except (ValueError, KeyError) as e:
            logger.warning(f"[CROSS] {framework2} cannot resolve {operator_name!r}: {e}")
            return self._empty_result(operator_name, framework1, framework2, mr, n_samples, "unsupported_f2")

        # 分别为两框架绑定 transform（若含 apply_operator 则需各自绑定）
        transform1 = MRPreChecker._bind_transform_code(mr.transform_code, op_f1)
        transform2 = MRPreChecker._bind_transform_code(mr.transform_code, op_f2)
        if transform1 is None or transform2 is None:
            logger.warning(f"[CROSS] Cannot bind transform for MR {mr.id[:8]}")
            return self._empty_result(operator_name, framework1, framework2, mr, n_samples, "transform_error")

        both_pass = both_fail = only_f1 = only_f2 = errors = 0
        output_diffs: List[float] = []

        for _ in range(n_samples):
            try:
                # 生成输入（使用 numpy 生成，再分别转换为各框架张量，保证数值相同）
                inputs_np = self.random_gen.generate(input_specs, backend2)
                # backend2 通常是 NumpyPlugin，生成 numpy 数组
                # backend1 需要 _to_tensor 转换

                x1_raw = backend1._to_tensor(backend2.to_numpy(inputs_np[0]))
                x2_raw = inputs_np[0]

                kwargs1 = MRPreChecker._build_kwargs([x1_raw])
                kwargs2 = MRPreChecker._build_kwargs([x2_raw])

                # Framework 1
                orig1 = op_f1(**kwargs1)
                trans_kwargs1 = transform1(kwargs1)
                trans1 = op_f1(**trans_kwargs1)
                oracle1 = self.verifier.verify(orig1, trans1, mr, backend1, x1_raw)

                # Framework 2
                orig2 = op_f2(**kwargs2)
                trans_kwargs2 = transform2(kwargs2)
                trans2 = op_f2(**trans_kwargs2)
                oracle2 = self.verifier.verify(orig2, trans2, mr, backend2, x2_raw)

                # 一致性分类
                p1, p2 = oracle1.passed, oracle2.passed
                if p1 and p2:
                    both_pass += 1
                elif p1 and not p2:
                    only_f1 += 1
                elif not p1 and p2:
                    only_f2 += 1
                else:
                    both_fail += 1

                # 输出数值差（f1 原始输出 vs f2 原始输出，相同输入）
                try:
                    out1_np = backend1.to_numpy(orig1).astype(float)
                    out2_np = backend2.to_numpy(orig2).astype(float)
                    if out1_np.shape == out2_np.shape:
                        diff = float(np.max(np.abs(out1_np - out2_np)))
                        output_diffs.append(diff)
                except Exception:
                    pass

            except Exception as e:
                errors += 1
                logger.debug(f"[CROSS] MR {mr.id[:8]} sample error: {e}")

        output_max_diff = max(output_diffs) if output_diffs else float("nan")
        output_mean_diff = float(np.mean(output_diffs)) if output_diffs else float("nan")
        output_close = (
            output_max_diff < self.OUTPUT_CLOSE_THRESHOLD
            if output_diffs else False
        )

        return CrossConsistencyResult(
            operator=operator_name,
            framework1=framework1,
            framework2=framework2,
            mr_id=mr.id,
            mr_description=mr.description,
            oracle_expr=mr.oracle_expr,
            n_samples=n_samples,
            both_pass=both_pass,
            only_f1_pass=only_f1,
            only_f2_pass=only_f2,
            both_fail=both_fail,
            errors=errors,
            output_max_diff=output_max_diff,
            output_mean_diff=output_mean_diff,
            output_close=output_close,
        )

    @staticmethod
    def _empty_result(
        operator: str, f1: str, f2: str,
        mr: MetamorphicRelation, n_samples: int, reason: str,
    ) -> CrossConsistencyResult:
        return CrossConsistencyResult(
            operator=operator, framework1=f1, framework2=f2,
            mr_id=mr.id, mr_description=mr.description, oracle_expr=mr.oracle_expr,
            n_samples=n_samples,
            both_pass=0, only_f1_pass=0, only_f2_pass=0, both_fail=0,
            errors=n_samples,
            output_max_diff=float("nan"), output_mean_diff=float("nan"),
            output_close=False,
        )
