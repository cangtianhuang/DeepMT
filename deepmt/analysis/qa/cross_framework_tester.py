"""
跨框架一致性测试器：验证不同框架在等价算子上的行为一致性。

设计目的：
  - 为论文 RQ3 提供跨框架一致性实验数据
  - 识别"合理的数值差异"和"可疑的行为不一致"
  - 验证同一条 MR 在不同框架中是否得出相同结论

核心概念：
  - 一致性（consistency）：两框架对同一输入执行同一 MR，结论相同（both pass 或 both fail）
  - 输出差异（output_diff）：两框架在相同输入上的实际数值差距，反映实现细节差异
  - 差异分类（diff_type）：对差异原因的细分（数值差异/形状不匹配/异常差异/行为分歧）

数据流：
  CrossFrameworkTester.compare_operator(op, f1, f2)
    → CrossConsistencyResult（每条 MR 一个）
    → CrossSessionResult（一次对比实验的完整记录）
    → 可选：save() 持久化到 data/results/cross_framework/<session_id>.json

框架支持：
  - "pytorch"：PyTorchPlugin（主链）
  - "numpy"：NumpyPlugin（数值参考后端，无需额外安装）
  - "paddlepaddle" / "paddle"：PaddlePlugin（真实第二框架）
  - 其他已注册框架：通过 PluginsManager.get_backend() 加载
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deepmt.analysis.verification.mr_prechecker import MRPreChecker
from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.analysis.verification.random_generator import RandomGenerator
from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.mr_generator.base.operator_catalog import OperatorCatalog
from deepmt.plugins.framework_plugin import FrameworkPlugin


# ── 差异类型定义 ─────────────────────────────────────────────────────────────


class DiffType:
    """
    跨框架差异类型常量。

    用于 CrossConsistencyResult.diff_type_counts 字典中的键，
    标识单次样本中观察到的差异原因。
    """

    NUMERIC_DIFF    = "numeric_diff"     # 两框架均成功，但输出数值差距超过阈值
    SHAPE_MISMATCH  = "shape_mismatch"   # 两框架输出形状不一致
    DTYPE_MISMATCH  = "dtype_mismatch"   # 两框架输出 dtype 不一致
    BEHAVIOR_DIFF   = "behavior_diff"    # MR 结论不一致（仅一个框架通过）
    EXCEPTION_F1    = "exception_f1"     # 仅 f1 抛出异常（f2 成功）
    EXCEPTION_F2    = "exception_f2"     # 仅 f2 抛出异常（f1 成功）
    BOTH_EXCEPTION  = "both_exception"   # 两框架均抛出异常


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class CrossConsistencyResult:
    """
    单条 MR 在两个框架之间的一致性对比结果。

    字段分四组：
      标识信息     — 算子、MR、两框架名称
      一致性统计   — 两框架结论是否一致（both_pass / only_f1_pass / only_f2_pass / both_fail）
      输出差异     — 相同输入下两框架实际数值差距
      差异分类统计 — 按 DiffType 分类的样本计数（H5 新增）
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

    # ── 差异分类统计（H5 新增） ───────────────────────────────────────────────
    # 每个键对应一种 DiffType，值为该类型出现的样本数（一个样本可属于多种类型）
    diff_type_counts: Dict[str, int] = field(default_factory=dict)

    # ── 失败样本明细（T4 新增） ───────────────────────────────────────────────
    # 记录有差异的样本供 case build 复现用，每个样本含输入摘要与两框架输出摘要
    failed_samples: List[Dict[str, Any]] = field(default_factory=list)

    # ── 静默数值差异指标（T6 新增） ───────────────────────────────────────────
    # consistency=1 但 output_close=False 的样本比例；silent_numeric_diff_rate>0 为可疑信号
    silent_numeric_diff_count: int = 0

    @property
    def silent_numeric_diff_rate(self) -> float:
        if self.total_valid == 0:
            return 0.0
        return self.silent_numeric_diff_count / self.total_valid

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
        d["silent_numeric_diff_rate"] = round(self.silent_numeric_diff_rate, 4)
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

    @property
    def silent_numeric_diff_count(self) -> int:
        return sum(r.silent_numeric_diff_count for r in self.mr_results)

    @property
    def silent_numeric_diff_rate(self) -> float:
        total = sum(r.total_valid for r in self.mr_results)
        return self.silent_numeric_diff_count / total if total > 0 else 0.0

    @property
    def diff_type_summary(self) -> Dict[str, int]:
        """汇总所有 MR 的差异类型计数（跨 MR 累加）。"""
        summary: Dict[str, int] = {}
        for r in self.mr_results:
            for k, v in r.diff_type_counts.items():
                summary[k] = summary.get(k, 0) + v
        return summary

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
            "silent_numeric_diff_count": self.silent_numeric_diff_count,
            "silent_numeric_diff_rate": round(self.silent_numeric_diff_rate, 4),
            "diff_type_summary": self.diff_type_summary,
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
      3. 差异类型分类（H5：shape/dtype/numeric/behavior/exception）

    用法示例：
        tester = CrossFrameworkTester()
        session = tester.compare_operator(
            "relu",
            framework1="pytorch",
            framework2="paddlepaddle",
            n_samples=20,
        )
        print(f"一致率: {session.overall_consistency_rate:.1%}")
        tester.save(session)

    CLI 等价：
        deepmt test cross relu
        deepmt test cross exp --framework1 pytorch --framework2 paddlepaddle
        deepmt test cross tanh --framework1 pytorch --framework2 numpy --json
    """

    DEFAULT_RESULTS_DIR = Path("data/results/cross_framework")
    OUTPUT_CLOSE_THRESHOLD = 1e-3  # 输出差 < 此阈值视为"数值接近"
    DEFAULT_KEEP_SAMPLES = 50       # 每条 MR 保留的失败样本上限
    SILENT_DIFF_WARN_RATE = 0.05    # silent_numeric_diff_rate 超过此值输出告警

    # 框架名称别名：支持用户输入 "paddle" 等简写
    _FRAMEWORK_ALIASES: Dict[str, str] = {
        "paddle": "paddlepaddle",
    }

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
        keep_samples: Optional[int] = None,
    ) -> CrossSessionResult:
        """
        对单个算子执行跨框架一致性测试。

        Args:
            operator_name: 算子名称（与 MR 知识库中的键一致）
            framework1:    第一框架（默认 pytorch）
            framework2:    第二框架（默认 numpy；可用 paddlepaddle/paddle）
            n_samples:     每条 MR 的测试样本数
            verified_only: 仅使用已验证的 MR

        Returns:
            CrossSessionResult，含每条 MR 的对比结果
        """
        framework1 = self._normalize_framework(framework1)
        framework2 = self._normalize_framework(framework2)

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

        keep = keep_samples if keep_samples is not None else self.DEFAULT_KEEP_SAMPLES
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
                keep_samples=keep,
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
        framework1 = self._normalize_framework(framework1)
        framework2 = self._normalize_framework(framework2)

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
            # 差异类型摘要
            diff_summary = s.diff_type_summary
            if diff_summary:
                diff_str = "  ".join(f"{k}={v}" for k, v in diff_summary.items() if v > 0)
                if diff_str:
                    lines.append(f"    差异类型: {diff_str}")
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

    def _normalize_framework(self, framework: str) -> str:
        """统一框架名称别名（如 paddle → paddlepaddle）。"""
        return self._FRAMEWORK_ALIASES.get(framework.lower(), framework.lower())

    def _get_backend(self, framework: str) -> FrameworkPlugin:
        """
        获取框架后端实例。

        支持的 framework 值：
          - "numpy"：NumpyPlugin（无需额外安装）
          - "paddlepaddle"：PaddlePlugin（需安装 paddlepaddle）
          - 其他已注册框架：通过 PluginsManager.get_backend() 加载
        """
        if framework.lower() == "numpy":
            from deepmt.plugins.numpy_plugin import NumpyPlugin
            return NumpyPlugin()
        if framework.lower() in ("paddlepaddle", "paddle"):
            from deepmt.plugins.paddle_plugin import PaddlePlugin
            return PaddlePlugin()
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
        keep_samples: int = 50,
    ) -> CrossConsistencyResult:
        """对单条 MR 执行跨框架对比，返回 CrossConsistencyResult（含差异类型分类）。"""

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
        silent_numeric_diff_count = 0
        output_diffs: List[float] = []
        diff_type_counts: Dict[str, int] = {}
        failed_samples: List[Dict[str, Any]] = []

        def _inc(key: str) -> None:
            diff_type_counts[key] = diff_type_counts.get(key, 0) + 1

        def _record_sample(
            seed: int, diff_type: str, x_raw: Any,
            orig1: Any, orig2: Any, numeric_diff: Optional[float] = None,
            f1_error: Optional[str] = None, f2_error: Optional[str] = None,
        ) -> None:
            if len(failed_samples) >= keep_samples:
                return
            from deepmt.analysis.reporting.evidence_collector import _summarize_tensor
            failed_samples.append({
                "seed": int(seed),
                "diff_type": diff_type,
                "input_summary": _summarize_tensor(x_raw),
                "f1_output_summary": _summarize_tensor(orig1) if orig1 is not None else None,
                "f2_output_summary": _summarize_tensor(orig2) if orig2 is not None else None,
                "numeric_diff": numeric_diff,
                "f1_error": f1_error,
                "f2_error": f2_error,
            })

        for sample_idx in range(n_samples):
            try:
                # 生成输入（使用 backend1 生成，再分别转换为各框架张量，保证数值相同）
                inputs_raw = self.random_gen.generate(input_specs, backend1)

                # 将 numpy/backend1 的输入转换为各自框架张量
                raw_np = backend1.to_numpy(inputs_raw[0])
                x1_raw = backend1._to_tensor(raw_np)
                x2_raw = backend2._to_tensor(raw_np)

                kwargs1 = MRPreChecker._build_kwargs([x1_raw])
                kwargs2 = MRPreChecker._build_kwargs([x2_raw])

                # ── Framework 1 执行 ──────────────────────────────────────
                f1_ok = False
                orig1 = trans1 = oracle1 = None
                f1_exc: Optional[Exception] = None
                try:
                    orig1 = op_f1(**kwargs1)
                    trans_kwargs1 = transform1(kwargs1)
                    trans1 = op_f1(**trans_kwargs1)
                    oracle1 = self.verifier.verify(orig1, trans1, mr, backend1, x1_raw)
                    f1_ok = True
                except Exception as e:
                    f1_exc = e

                # ── Framework 2 执行 ──────────────────────────────────────
                f2_ok = False
                orig2 = trans2 = oracle2 = None
                f2_exc: Optional[Exception] = None
                try:
                    orig2 = op_f2(**kwargs2)
                    trans_kwargs2 = transform2(kwargs2)
                    trans2 = op_f2(**trans_kwargs2)
                    oracle2 = self.verifier.verify(orig2, trans2, mr, backend2, x2_raw)
                    f2_ok = True
                except Exception as e:
                    f2_exc = e

                # ── 异常分类 ──────────────────────────────────────────────
                if not f1_ok and not f2_ok:
                    errors += 1
                    _inc(DiffType.BOTH_EXCEPTION)
                    _record_sample(sample_idx, DiffType.BOTH_EXCEPTION, x1_raw,
                                   None, None,
                                   f1_error=repr(f1_exc), f2_error=repr(f2_exc))
                    continue
                if not f1_ok:
                    errors += 1
                    _inc(DiffType.EXCEPTION_F1)
                    _record_sample(sample_idx, DiffType.EXCEPTION_F1, x1_raw,
                                   None, orig2, f1_error=repr(f1_exc))
                    continue
                if not f2_ok:
                    errors += 1
                    _inc(DiffType.EXCEPTION_F2)
                    _record_sample(sample_idx, DiffType.EXCEPTION_F2, x1_raw,
                                   orig1, None, f2_error=repr(f2_exc))
                    continue

                # ── 一致性分类 ────────────────────────────────────────────
                p1, p2 = oracle1.passed, oracle2.passed
                consistent = (p1 == p2)
                if p1 and p2:
                    both_pass += 1
                elif p1 and not p2:
                    only_f1 += 1
                    _inc(DiffType.BEHAVIOR_DIFF)
                elif not p1 and p2:
                    only_f2 += 1
                    _inc(DiffType.BEHAVIOR_DIFF)
                else:
                    both_fail += 1

                # ── 输出形状/dtype 差异分类 ───────────────────────────────
                sample_diff_types: List[str] = []
                sample_numeric_diff: Optional[float] = None
                try:
                    shape1 = backend1.get_shape(orig1)
                    shape2 = backend2.get_shape(orig2)
                    if shape1 != shape2:
                        _inc(DiffType.SHAPE_MISMATCH)
                        sample_diff_types.append(DiffType.SHAPE_MISMATCH)

                    out1_np = backend1.to_numpy(orig1)
                    out2_np = backend2.to_numpy(orig2)

                    # dtype 差异（仅 kind 不同时才计入，如 float32 vs int64）
                    if (hasattr(out1_np, "dtype") and hasattr(out2_np, "dtype")
                            and out1_np.dtype.kind != out2_np.dtype.kind):
                        _inc(DiffType.DTYPE_MISMATCH)
                        sample_diff_types.append(DiffType.DTYPE_MISMATCH)

                    # 数值差异
                    if shape1 == shape2:
                        out1_f = out1_np.astype(float)
                        out2_f = out2_np.astype(float)
                        diff = float(np.max(np.abs(out1_f - out2_f)))
                        output_diffs.append(diff)
                        sample_numeric_diff = diff
                        if diff >= self.OUTPUT_CLOSE_THRESHOLD:
                            _inc(DiffType.NUMERIC_DIFF)
                            sample_diff_types.append(DiffType.NUMERIC_DIFF)
                            # silent：两侧结论一致但数值差异超阈值
                            if consistent:
                                silent_numeric_diff_count += 1
                except Exception:
                    pass

                # ── 记录失败样本（仅 behavior_diff 或显著差异） ──────────
                if not consistent:
                    _record_sample(
                        sample_idx, DiffType.BEHAVIOR_DIFF, x1_raw,
                        orig1, orig2, numeric_diff=sample_numeric_diff,
                    )
                elif sample_diff_types:
                    _record_sample(
                        sample_idx, sample_diff_types[0], x1_raw,
                        orig1, orig2, numeric_diff=sample_numeric_diff,
                    )

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
            diff_type_counts=diff_type_counts,
            failed_samples=failed_samples,
            silent_numeric_diff_count=silent_numeric_diff_count,
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
            diff_type_counts={reason: n_samples},
        )
