"""模型层测试执行器：加载模型基准集，执行 MR 蜕变测试，持久化结果。

执行主链：
  1. 从 ModelBenchmarkRegistry 加载模型（含 model_instance）
  2. 通过 ModelMRGenerator 生成（或直接接受传入的）MR 列表
  3. 生成随机输入（float / int 根据 model_type 自动选择）
  4. 执行原始 + 变换后前向推理，捕获输出
  5. 通过 ModelVerifier 验证 oracle
  6. 将结果存入 ResultsManager（model 层）
  7. 返回 ModelTestSummary

与算子层 BatchTestRunner 的对比：
  - 算子层：输入通过 RandomGenerator + input_specs 生成，算子用 _resolve_operator 解析
  - 模型层：输入由本模块直接生成（torch.randn / torch.randint），模型实例直接调用
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deepmt.analysis.model_verifier import ModelVerifier
from deepmt.benchmarks.models.model_registry import ModelBenchmarkRegistry
from deepmt.core.logger import logger
from deepmt.core.results_manager import ResultsManager
from deepmt.ir.schema import MetamorphicRelation, ModelIR, OracleResult
from deepmt.model.graph_analyzer import ModelGraphAnalyzer
from deepmt.mr_generator.model.model_mr_generator import ModelMRGenerator

# 懒导入 torch
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("模型层测试需要 PyTorch，请先安装：pip install torch")
    return _torch


# ── 结果数据结构 ───────────────────────────────────────────────────────────────


@dataclass
class ModelMRTestSummary:
    """单条 MR 在一个模型上的测试摘要。"""

    mr_id: str
    description: str
    oracle_expr: str
    passed: int
    failed: int
    errors: int

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class ModelTestSummary:
    """单个模型批量测试的汇总结果。"""

    model_name: str
    model_type: str
    framework: str
    mr_count: int
    n_samples: int
    total_cases: int
    passed: int
    failed: int
    errors: int
    mr_summaries: List[ModelMRTestSummary] = field(default_factory=list)
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)  # 失败样本摘要

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "framework": self.framework,
            "mr_count": self.mr_count,
            "n_samples": self.n_samples,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 4),
            "mr_summaries": [
                {
                    "mr_id": m.mr_id,
                    "description": m.description,
                    "oracle_expr": m.oracle_expr,
                    "passed": m.passed,
                    "failed": m.failed,
                    "errors": m.errors,
                    "total": m.total,
                    "pass_rate": round(m.pass_rate, 4),
                }
                for m in self.mr_summaries
            ],
            "failure_cases": self.failure_cases[:10],  # 只输出前 10 个失败案例
        }


# ── 输入生成 ───────────────────────────────────────────────────────────────────


def _generate_input(model_ir: ModelIR, batch_size: int = 4) -> Any:
    """根据 ModelIR 的 input_shape 和 model_type 生成随机输入。

    - float 输入（mlp/cnn）：torch.randn(batch, *input_shape)
    - int 输入（rnn/transformer）：torch.randint(0, vocab_size, (batch, seq_len))
    """
    torch = _get_torch()
    input_shape = model_ir.input_shape or (64,)
    input_dtype = model_ir.metadata.get("input_dtype", "float32")

    if input_dtype == "int64":
        vocab_size = model_ir.metadata.get("vocab_size", 100)
        return torch.randint(0, vocab_size, (batch_size, *input_shape))
    else:
        return torch.randn(batch_size, *input_shape)


def _apply_transform(transform_code: str, x: Any) -> Optional[Any]:
    """对输入张量应用 transform_code（lambda x: ...）。"""
    try:
        transform = eval(transform_code)  # noqa: S307
        result = transform(x)
        return result
    except Exception as e:
        logger.debug(f"[ModelTestRunner] transform 执行失败: {e}")
        return None


# ── 测试执行器 ────────────────────────────────────────────────────────────────


class ModelTestRunner:
    """模型层测试执行器。

    用法::

        runner = ModelTestRunner()
        # 测试单个模型（使用基准注册表 + 自动生成 MR）
        summary = runner.run_model("SimpleMLP", n_samples=20)
        print(summary.pass_rate)

        # 批量测试所有基准模型
        summaries = runner.run_all(n_samples=10)
    """

    def __init__(
        self,
        benchmark_registry: Optional[ModelBenchmarkRegistry] = None,
        mr_generator: Optional[ModelMRGenerator] = None,
        verifier: Optional[ModelVerifier] = None,
        results_manager: Optional[ResultsManager] = None,
        max_failure_cases: int = 10,
    ):
        self.benchmark_registry = benchmark_registry or ModelBenchmarkRegistry()
        self.mr_generator = mr_generator or ModelMRGenerator()
        self.verifier = verifier or ModelVerifier()
        self.results_manager = results_manager or ResultsManager()
        self.max_failure_cases = max_failure_cases

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def run_model(
        self,
        model_name: str,
        n_samples: int = 10,
        mrs: Optional[List[MetamorphicRelation]] = None,
        max_mrs: Optional[int] = None,
        batch_size: int = 4,
    ) -> ModelTestSummary:
        """测试单个基准模型。

        Args:
            model_name:  基准模型名称（如 "SimpleMLP"）
            n_samples:   随机测试样本数
            mrs:         若指定则使用给定 MR 列表，否则自动生成
            max_mrs:     自动生成时的最大 MR 数量
            batch_size:  每次推理的 batch 大小

        Returns:
            ModelTestSummary
        """
        model_ir = self.benchmark_registry.get(model_name, with_instance=True)
        if model_ir is None:
            logger.error(f"[ModelTestRunner] 未找到模型: {model_name!r}")
            return self._empty_summary(model_name, "unknown", n_samples)

        return self.run_model_ir(
            model_ir,
            n_samples=n_samples,
            mrs=mrs,
            max_mrs=max_mrs,
            batch_size=batch_size,
        )

    def run_model_ir(
        self,
        model_ir: ModelIR,
        n_samples: int = 10,
        mrs: Optional[List[MetamorphicRelation]] = None,
        max_mrs: Optional[int] = None,
        batch_size: int = 4,
    ) -> ModelTestSummary:
        """直接使用 ModelIR 对象执行测试（支持外部传入自定义模型）。

        Args:
            model_ir:    含 model_instance 的 ModelIR
            n_samples:   随机测试样本数
            mrs:         若指定则使用给定 MR 列表
            max_mrs:     自动生成时的最大 MR 数量
            batch_size:  每次推理的 batch 大小
        """
        if model_ir.model_instance is None:
            logger.error(f"[ModelTestRunner] {model_ir.name} 的 model_instance 为 None")
            return self._empty_summary(model_ir.name, model_ir.model_type, n_samples)

        # 生成 MR
        if mrs is None:
            mrs = self.mr_generator.generate(model_ir, max_per_model=max_mrs)
        if not mrs:
            logger.warning(f"[ModelTestRunner] {model_ir.name}: 无可用 MR")
            return self._empty_summary(model_ir.name, model_ir.model_type, n_samples)

        framework = model_ir.framework or "pytorch"
        logger.info(
            f"[ModelTestRunner] {model_ir.name}: "
            f"{len(mrs)} MRs × {n_samples} samples"
        )

        mr_summaries: List[ModelMRTestSummary] = []
        failure_cases: List[Dict[str, Any]] = []
        total_passed = total_failed = total_errors = 0

        for mr in mrs:
            mr_summary, mr_failures = self._run_single_mr(
                model_ir=model_ir,
                mr=mr,
                n_samples=n_samples,
                batch_size=batch_size,
            )
            mr_summaries.append(mr_summary)
            total_passed += mr_summary.passed
            total_failed += mr_summary.failed
            total_errors += mr_summary.errors
            failure_cases.extend(mr_failures)

        # 持久化：将模型层结果写入 results_manager
        self._persist_results(model_ir, mrs, mr_summaries, framework)

        summary = ModelTestSummary(
            model_name=model_ir.name,
            model_type=model_ir.model_type,
            framework=framework,
            mr_count=len(mrs),
            n_samples=n_samples,
            total_cases=total_passed + total_failed,
            passed=total_passed,
            failed=total_failed,
            errors=total_errors,
            mr_summaries=mr_summaries,
            failure_cases=failure_cases[:self.max_failure_cases],
        )
        logger.info(
            f"[ModelTestRunner] {model_ir.name}: "
            f"passed={summary.passed}/{summary.total_cases} "
            f"errors={summary.errors}"
        )
        return summary

    def run_all(
        self,
        framework: str = "pytorch",
        n_samples: int = 10,
        max_mrs: Optional[int] = None,
        batch_size: int = 4,
    ) -> List[ModelTestSummary]:
        """批量测试所有基准模型。

        Args:
            framework:  框架名称过滤
            n_samples:  每条 MR 的测试样本数
            max_mrs:    每个模型最多生成的 MR 数
            batch_size: 每次推理的 batch 大小

        Returns:
            每个模型对应一个 ModelTestSummary 的列表
        """
        model_names = self.benchmark_registry.names(framework=framework)
        summaries = []
        for name in model_names:
            s = self.run_model(
                name,
                n_samples=n_samples,
                max_mrs=max_mrs,
                batch_size=batch_size,
            )
            summaries.append(s)
        return summaries

    # ── 内部实现 ──────────────────────────────────────────────────────────────

    def _run_single_mr(
        self,
        model_ir: ModelIR,
        mr: MetamorphicRelation,
        n_samples: int,
        batch_size: int,
    ) -> Tuple[ModelMRTestSummary, List[Dict[str, Any]]]:
        """对单条 MR 运行 n_samples 次测试。

        Returns:
            (ModelMRTestSummary, failure_cases_list)
        """
        torch = _get_torch()
        model = model_ir.model_instance
        passed = failed = errors = 0
        failure_cases: List[Dict[str, Any]] = []

        for i in range(n_samples):
            try:
                with torch.no_grad():
                    x = _generate_input(model_ir, batch_size=batch_size)
                    orig_output = model(x)

                    x_trans = _apply_transform(mr.transform_code, x)
                    if x_trans is None:
                        errors += 1
                        continue

                    trans_output = model(x_trans)

                oracle_result = self.verifier.verify(orig_output, trans_output, mr)

                if oracle_result.passed:
                    passed += 1
                else:
                    failed += 1
                    if len(failure_cases) < self.max_failure_cases:
                        failure_cases.append({
                            "model": model_ir.name,
                            "mr_id": mr.id,
                            "mr_description": mr.description,
                            "oracle_expr": mr.oracle_expr,
                            "sample_idx": i,
                            "detail": oracle_result.detail,
                            "actual_diff": oracle_result.actual_diff,
                        })
                    logger.debug(
                        f"[ModelTestRunner] {mr.id[:8]} sample {i+1} FAIL: "
                        f"{oracle_result.detail}"
                    )

            except Exception as e:
                errors += 1
                logger.debug(
                    f"[ModelTestRunner] {mr.id[:8]} sample {i+1} ERROR: {e}"
                )

        summary = ModelMRTestSummary(
            mr_id=mr.id,
            description=mr.description,
            oracle_expr=mr.oracle_expr,
            passed=passed,
            failed=failed,
            errors=errors,
        )
        return summary, failure_cases

    def _persist_results(
        self,
        model_ir: ModelIR,
        mrs: List[MetamorphicRelation],
        mr_summaries: List[ModelMRTestSummary],
        framework: str,
    ) -> None:
        """将模型层测试摘要写入 ResultsManager（使用 model 层前缀）。"""
        try:
            # 将模型测试结果以 oracle_results 格式写入
            # 使用 OperatorIR 接口以兼容现有 ResultsManager（模型名作为 api_path）
            from deepmt.ir.schema import OperatorIR

            op_ir = OperatorIR(
                name=f"model:{model_ir.name}",
                api_path=model_ir.name,
            )
            oracle_results_flat = []
            for mr, mr_sum in zip(mrs, mr_summaries):
                for _ in range(mr_sum.passed):
                    oracle_results_flat.append((
                        mr,
                        OracleResult(
                            passed=True,
                            expr=mr.oracle_expr,
                            actual_diff=0.0,
                            tolerance=0.0,
                        ),
                    ))
                for _ in range(mr_sum.failed):
                    oracle_results_flat.append((
                        mr,
                        OracleResult(
                            passed=False,
                            expr=mr.oracle_expr,
                            actual_diff=float("nan"),
                            tolerance=0.0,
                            detail="MODEL_LAYER_FAILURE",
                        ),
                    ))
            if oracle_results_flat:
                self.results_manager.store_result(op_ir, oracle_results_flat, framework)
        except Exception as e:
            logger.debug(f"[ModelTestRunner] 持久化失败（不影响主流程）: {e}")

    @staticmethod
    def _empty_summary(
        model_name: str, model_type: str, n_samples: int
    ) -> ModelTestSummary:
        return ModelTestSummary(
            model_name=model_name,
            model_type=model_type,
            framework="pytorch",
            mr_count=0,
            n_samples=n_samples,
            total_cases=0,
            passed=0,
            failed=0,
            errors=0,
        )
