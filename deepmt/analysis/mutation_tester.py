"""
变异测试器：通过注入已知错误的算子实现，验证 BatchTestRunner 的缺陷检测能力。

设计目的：
  - 在没有真实框架缺陷的情况下，用"显然错误"的变异体验证检测管道的完整性
  - 受控评估：若系统无法发现已知缺陷，说明 MR / oracle 或执行链路存在问题
  - 为 Phase D 的论文 RQ2（缺陷检测率）提供受控实验数据

变异类型（MutantType）：
  NEGATE_OUTPUT   — 取反输出：f(x) = -real_f(x)，破坏所有等值关系
  ADD_CONSTANT    — 添加偏置：f(x) = real_f(x) + C，破坏等值和线性关系
  SCALE_WRONG     — 错误缩放：f(x) = k·real_f(x)，当 k 与 MR 期望比例不符时被检出
  IDENTITY        — 恒等函数：f(x) = x，破坏非线性算子的输出关系
  ZERO_OUTPUT     — 恒零输出：f(x) = zeros_like(real_f(x))，破坏非零性质

使用方式：
  # 验证 relu 的 MR 能否检测到"输出取反"的变异
  tester = MutationTester()
  result = tester.run("torch.nn.functional.relu", MutantType.NEGATE_OUTPUT, "pytorch")
  print(result.detection_rate)  # 期望 > 0（被检出）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from deepmt.core.logger import logger
from deepmt.core.plugins_manager import FrameworkType, get_plugins_manager
from deepmt.engine.batch_test_runner import BatchTestRunner, OperatorTestSummary
from deepmt.ir.schema import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository


# ── 变异类型枚举 ──────────────────────────────────────────────────────────────


class MutantType(str, Enum):
    """预定义变异类型。"""

    NEGATE_OUTPUT = "negate"       # -f(x)：取反输出
    ADD_CONSTANT = "add_const"     # f(x) + C：添加常数偏置（默认 C=1.0）
    SCALE_WRONG = "scale"          # k·f(x)：错误缩放系数（默认 k=2.0）
    IDENTITY = "identity"          # x：直接返回输入，忽略算子
    ZERO_OUTPUT = "zero"           # 0：始终返回零张量


# ── 变异结果数据类 ─────────────────────────────────────────────────────────────


@dataclass
class MutantResult:
    """单次变异测试的结果。"""

    operator_name: str
    mutant_type: MutantType
    framework: str
    n_samples: int
    mr_count: int
    detected_cases: int    # 失败（被检出）的样本数
    total_cases: int        # 执行成功的样本总数（通过 + 失败）
    errors: int
    mr_details: List[Dict] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """检出率：detected_cases / total_cases。total=0 时返回 0.0。"""
        return self.detected_cases / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def detected(self) -> bool:
        """变异是否被至少一个测试案例检出。"""
        return self.detected_cases > 0

    def to_dict(self) -> Dict:
        return {
            "operator": self.operator_name,
            "mutant_type": self.mutant_type.value,
            "framework": self.framework,
            "n_samples": self.n_samples,
            "mr_count": self.mr_count,
            "detected_cases": self.detected_cases,
            "total_cases": self.total_cases,
            "errors": self.errors,
            "detection_rate": round(self.detection_rate, 4),
            "detected": self.detected,
            "mr_details": self.mr_details,
        }


# ── 变异函数工厂 ──────────────────────────────────────────────────────────────


def create_mutant_func(
    real_func: Callable,
    mutant_type: MutantType,
    scale: float = 2.0,
    const: float = 1.0,
) -> Callable:
    """
    基于真实算子函数，创建对应变异类型的包装函数。

    Args:
        real_func:   真实算子函数（作为基准，某些变异类型会调用它）
        mutant_type: 变异类型
        scale:       SCALE_WRONG 时使用的缩放系数（默认 2.0）
        const:       ADD_CONSTANT 时使用的偏置值（默认 1.0）

    Returns:
        具有相同 **kwargs 签名的变异可调用对象
    """
    if mutant_type == MutantType.NEGATE_OUTPUT:
        def mutant(**kwargs):
            return -real_func(**kwargs)

    elif mutant_type == MutantType.ADD_CONSTANT:
        _c = float(const)
        def mutant(**kwargs):
            result = real_func(**kwargs)
            return result + _c

    elif mutant_type == MutantType.SCALE_WRONG:
        _k = float(scale)
        def mutant(**kwargs):
            return _k * real_func(**kwargs)

    elif mutant_type == MutantType.IDENTITY:
        def mutant(**kwargs):
            # 直接返回主输入张量，忽略算子语义
            inp = kwargs.get("input", next(iter(kwargs.values())))
            return inp

    elif mutant_type == MutantType.ZERO_OUTPUT:
        def mutant(**kwargs):
            result = real_func(**kwargs)
            try:
                import torch
                return torch.zeros_like(result)
            except Exception:
                return result * 0

    else:
        raise ValueError(f"Unknown MutantType: {mutant_type!r}")

    mutant.__name__ = f"mutant_{mutant_type}"
    return mutant


# ── 变异测试器 ────────────────────────────────────────────────────────────────


class MutationTester:
    """
    变异测试器：将变异函数注入 BatchTestRunner，验证 MR 的缺陷检测能力。

    主要用途：
      1. 受控评估（Phase D4）：测量 MR 对已知变异的检出率
      2. 管道验证：确认 oracle 表达式与执行链路工作正常
      3. MR 质量评估：筛选出对变异敏感的高质量 MR

    用法示例：
        tester = MutationTester()
        result = tester.run(
            "torch.nn.functional.relu",
            MutantType.NEGATE_OUTPUT,
            "pytorch",
            n_samples=10,
        )
        print(f"检出率: {result.detection_rate:.1%}")
        assert result.detected, "变异未被检出，MR 或管道存在问题！"
    """

    def __init__(
        self,
        repo: Optional[MRRepository] = None,
        runner: Optional[BatchTestRunner] = None,
    ):
        self.repo = repo or MRRepository()
        # 变异测试不存储结果（用 mock ResultsManager）
        if runner is None:
            from deepmt.core.results_manager import ResultsManager
            from unittest.mock import MagicMock
            mock_rm = MagicMock(spec=ResultsManager)
            self.runner = BatchTestRunner(repo=self.repo, results_manager=mock_rm)
        else:
            self.runner = runner

    def run(
        self,
        operator_name: str,
        mutant_type: MutantType,
        framework: FrameworkType,
        n_samples: int = 10,
        verified_only: bool = False,
        scale: float = 2.0,
        const: float = 1.0,
        mr_id: Optional[str] = None,
    ) -> MutantResult:
        """
        对单个算子注入变异并执行测试。

        Args:
            operator_name: 目标算子（MR 知识库中的键）
            mutant_type:   变异类型
            framework:     目标框架
            n_samples:     每条 MR 的测试样本数
            verified_only: 仅使用已验证 MR
            scale:         SCALE_WRONG 时的缩放系数
            const:         ADD_CONSTANT 时的偏置值
            mr_id:         若指定则只测试该 MR

        Returns:
            MutantResult，含检出率和逐 MR 详情
        """
        fw_str = str(framework)
        logger.info(
            f"[MUTATE] {operator_name} | mutant={mutant_type} | framework={fw_str}"
        )

        # 解析真实算子函数
        try:
            backend = get_plugins_manager().get_backend(framework)
            real_func = backend._resolve_operator(operator_name)
        except (KeyError, ValueError) as e:
            logger.error(f"[MUTATE] Cannot resolve operator {operator_name!r}: {e}")
            return MutantResult(
                operator_name=operator_name,
                mutant_type=mutant_type,
                framework=fw_str,
                n_samples=n_samples,
                mr_count=0,
                detected_cases=0,
                total_cases=0,
                errors=0,
            )

        # 创建变异函数
        mutant_func = create_mutant_func(real_func, mutant_type, scale=scale, const=const)

        # 通过 BatchTestRunner 执行（注入变异函数）
        summary: OperatorTestSummary = self.runner.run_operator(
            operator_name=operator_name,
            framework=framework,
            n_samples=n_samples,
            verified_only=verified_only,
            mr_id=mr_id,
            operator_func=mutant_func,
        )

        # 在变异测试中，"failed" = "被检出"
        mr_details = [
            {
                "mr_id": m.mr_id,
                "description": m.description,
                "detected_cases": m.failed,  # 失败 = 检出变异
                "total_cases": m.total,
                "errors": m.errors,
                "detection_rate": round(m.failed / m.total if m.total > 0 else 0.0, 4),
            }
            for m in summary.mr_summaries
        ]

        return MutantResult(
            operator_name=operator_name,
            mutant_type=mutant_type,
            framework=fw_str,
            n_samples=n_samples,
            mr_count=summary.mr_count,
            detected_cases=summary.failed,
            total_cases=summary.total_cases,
            errors=summary.errors,
            mr_details=mr_details,
        )

    def run_all_mutants(
        self,
        operator_name: str,
        framework: FrameworkType,
        mutant_types: Optional[List[MutantType]] = None,
        n_samples: int = 10,
        verified_only: bool = False,
    ) -> List[MutantResult]:
        """
        对一个算子运行所有（或指定）变异类型，返回各变异的检出结果列表。

        Args:
            operator_name: 目标算子
            framework:     目标框架
            mutant_types:  变异类型列表；为 None 时运行全部 MutantType
            n_samples:     每条 MR 的测试样本数
            verified_only: 仅使用已验证 MR

        Returns:
            每种变异类型对应一个 MutantResult 的列表
        """
        types = mutant_types or list(MutantType)
        results = []
        for mt in types:
            result = self.run(
                operator_name=operator_name,
                mutant_type=mt,
                framework=framework,
                n_samples=n_samples,
                verified_only=verified_only,
            )
            results.append(result)
        return results
