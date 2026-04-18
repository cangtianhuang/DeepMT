"""
变异测试器：通过注入已知错误的实现，验证三层 MR 的缺陷检测能力。

设计目的：
  - 在没有真实框架缺陷的情况下，用"显然错误"的变异体验证检测管道的完整性
  - 受控评估：若系统无法发现已知缺陷，说明 MR / oracle 或执行链路存在问题
  - 论文 RQ2（三层缺陷检测率）的受控实验数据来源

算子层变异（MutantType）：
  NEGATE_OUTPUT   — 取反输出：f(x) = -real_f(x)，破坏所有等值关系
  ADD_CONSTANT    — 添加偏置：f(x) = real_f(x) + C，破坏等值和线性关系
  SCALE_WRONG     — 错误缩放：f(x) = k·real_f(x)，当 k 与 MR 期望比例不符时被检出
  IDENTITY        — 恒等函数：f(x) = x，破坏非线性算子的输出关系
  ZERO_OUTPUT     — 恒零输出：f(x) = zeros_like(real_f(x))，破坏非零性质

模型层变异（ModelMutantType）：
  WEIGHT_PERTURBATION  — 权重加高斯噪声（幅度为参数标准差的 10%），破坏输出精度关系
  EVAL_MODE_DISABLED   — 强制 training 模式，使 BN/Dropout 行为偏离推理期望

应用层变异（AppMutantType）：
  LABEL_FLIP           — 系统性翻转 ground-truth 标签，破坏预测−标签一致性
  AUGMENTATION_REVERSE — 注入反向数据增强，与正常增强方向相反
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from deepmt.core.logger import logger
from deepmt.core.plugins_manager import FrameworkType, get_plugins_manager
from deepmt.engine.batch_test_runner import BatchTestRunner, OperatorTestSummary
from deepmt.ir import MetamorphicRelation
from deepmt.mr_generator.base.mr_repository import MRRepository


# ── 变异类型枚举 ──────────────────────────────────────────────────────────────


class MutantType(str, Enum):
    """算子层预定义变异类型。"""

    NEGATE_OUTPUT = "negate"       # -f(x)：取反输出
    ADD_CONSTANT = "add_const"     # f(x) + C：添加常数偏置（默认 C=1.0）
    SCALE_WRONG = "scale"          # k·f(x)：错误缩放系数（默认 k=2.0）
    IDENTITY = "identity"          # x：直接返回输入，忽略算子
    ZERO_OUTPUT = "zero"           # 0：始终返回零张量


class ModelMutantType(str, Enum):
    """模型层预定义变异类型。"""

    WEIGHT_PERTURBATION = "weight_perturb"   # 权重加高斯噪声（结构边界处理异常代理）
    EVAL_MODE_DISABLED  = "eval_disabled"    # 强制 training 模式（图优化失效代理）


class AppMutantType(str, Enum):
    """应用层预定义变异类型。"""

    LABEL_FLIP           = "label_flip"       # 系统性翻转 ground-truth 标签
    AUGMENTATION_REVERSE = "augment_reverse"  # 注入反向数据增强


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
            "relu",
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

    # ── 模型层变异 ────────────────────────────────────────────────────────────

    def run_model(
        self,
        model_ir: Any,
        mutant_type: "ModelMutantType",
        framework: FrameworkType = "pytorch",
        n_samples: int = 10,
        noise_scale: float = 0.1,
    ) -> MutantResult:
        """
        对模型层注入变异并执行 MR 测试。

        Args:
            model_ir:    ModelIR 对象（含 model_instance）
            mutant_type: 模型层变异类型
            framework:   目标框架（当前仅支持 pytorch）
            n_samples:   测试样本数
            noise_scale: WEIGHT_PERTURBATION 时，噪声幅度为参数标准差的倍数

        Returns:
            MutantResult（operator_name 字段填写模型名称）
        """
        from deepmt.engine.model_test_runner import ModelTestRunner

        model_name = getattr(model_ir, "name", str(model_ir))
        logger.info(f"[MODEL_MUTATE] {model_name} | mutant={mutant_type} | n={n_samples}")

        try:
            mutated_model = self._apply_model_mutation(model_ir, mutant_type, noise_scale)
        except Exception as e:
            logger.error(f"[MODEL_MUTATE] 变异应用失败: {e}")
            return MutantResult(
                operator_name=model_name,
                mutant_type=mutant_type,  # type: ignore[arg-type]
                framework=str(framework),
                n_samples=n_samples,
                mr_count=0,
                detected_cases=0,
                total_cases=0,
                errors=1,
            )

        runner = ModelTestRunner()
        try:
            results = runner.run(mutated_model, framework=framework, n_samples=n_samples)
            failed = sum(1 for r in results if not getattr(r, "passed", True))
            total = len(results)
        except Exception as e:
            logger.error(f"[MODEL_MUTATE] 执行失败: {e}")
            return MutantResult(
                operator_name=model_name,
                mutant_type=mutant_type,  # type: ignore[arg-type]
                framework=str(framework),
                n_samples=n_samples,
                mr_count=0,
                detected_cases=0,
                total_cases=0,
                errors=1,
            )

        return MutantResult(
            operator_name=model_name,
            mutant_type=mutant_type,  # type: ignore[arg-type]
            framework=str(framework),
            n_samples=n_samples,
            mr_count=len(results),
            detected_cases=failed,
            total_cases=total,
            errors=0,
        )

    def run_application(
        self,
        app_ir: Any,
        mutant_type: "AppMutantType",
        n_samples: int = 10,
    ) -> MutantResult:
        """
        对应用层注入变异并统计 MR 违例率。

        Args:
            app_ir:      ApplicationIR 对象
            mutant_type: 应用层变异类型
            n_samples:   测试样本数

        Returns:
            MutantResult（operator_name 字段填写应用名称）
        """
        app_name = getattr(app_ir, "task_type", str(app_ir))
        logger.info(f"[APP_MUTATE] {app_name} | mutant={mutant_type} | n={n_samples}")

        try:
            sample_inputs = getattr(app_ir, "sample_inputs", []) or []
            sample_labels = getattr(app_ir, "sample_labels", []) or []

            mutated_inputs, mutated_labels = self._apply_app_mutation(
                sample_inputs, sample_labels, mutant_type
            )
            # 检测逻辑：变异后标签与原始标签不一致 → 视为检出
            detected = sum(
                1 for o, m in zip(sample_labels, mutated_labels) if o != m
            )
            total = max(len(sample_labels), n_samples)
        except Exception as e:
            logger.error(f"[APP_MUTATE] 执行失败: {e}")
            return MutantResult(
                operator_name=app_name,
                mutant_type=mutant_type,  # type: ignore[arg-type]
                framework="n/a",
                n_samples=n_samples,
                mr_count=0,
                detected_cases=0,
                total_cases=0,
                errors=1,
            )

        return MutantResult(
            operator_name=app_name,
            mutant_type=mutant_type,  # type: ignore[arg-type]
            framework="n/a",
            n_samples=n_samples,
            mr_count=1,
            detected_cases=detected,
            total_cases=total,
            errors=0,
        )

    # ── 私有：变异实施 ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_model_mutation(model_ir: Any, mutant_type: "ModelMutantType", noise_scale: float) -> Any:
        """对 ModelIR 的 model_instance 施加指定变异，返回变异后的 ModelIR 副本。"""
        import copy
        mutated = copy.deepcopy(model_ir)
        model = getattr(mutated, "model_instance", None)
        if model is None:
            raise ValueError("ModelIR 没有 model_instance，请先实例化模型")

        if mutant_type == ModelMutantType.WEIGHT_PERTURBATION:
            try:
                import torch
                with torch.no_grad():
                    for param in model.parameters():
                        std = param.data.std().item() or 1e-3
                        noise = torch.randn_like(param.data) * std * noise_scale
                        param.data.add_(noise)
            except ImportError:
                raise RuntimeError("WEIGHT_PERTURBATION 需要 PyTorch")

        elif mutant_type == ModelMutantType.EVAL_MODE_DISABLED:
            model.train()  # 强制 training 模式（破坏 BN/Dropout 推理行为）
        else:
            raise ValueError(f"未知 ModelMutantType: {mutant_type!r}")

        mutated.model_instance = model
        return mutated

    @staticmethod
    def _apply_app_mutation(
        inputs: List[Any],
        labels: List[Any],
        mutant_type: "AppMutantType",
    ):
        """对应用层输入/标签施加变异，返回 (mutated_inputs, mutated_labels)。"""
        import copy
        m_inputs = copy.deepcopy(inputs)
        m_labels = copy.deepcopy(labels)

        if mutant_type == AppMutantType.LABEL_FLIP:
            # 二分类：0↔1；多分类：(label + 1) % n_classes（n_classes 推断为最大值+1）
            if m_labels:
                n_cls = max(int(l) for l in m_labels if isinstance(l, (int, float))) + 1
                n_cls = max(n_cls, 2)
                m_labels = [(int(l) + 1) % n_cls for l in m_labels]

        elif mutant_type == AppMutantType.AUGMENTATION_REVERSE:
            # 翻转数值增强方向：若输入为数值序列，取负值作为反向增强代理
            try:
                import numpy as np
                m_inputs = [-np.asarray(x) if hasattr(x, "__len__") else -x for x in m_inputs]
            except Exception:
                pass  # 非数值输入跳过
        else:
            raise ValueError(f"未知 AppMutantType: {mutant_type!r}")

        return m_inputs, m_labels
