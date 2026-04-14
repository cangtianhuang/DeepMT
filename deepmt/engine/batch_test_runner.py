"""
批量测试执行器：从 MR 知识库选取算子和 MR，使用 RandomGenerator 自动生成输入执行测试。

执行主链：
  1. 从 MR 知识库读取算子列表（含已生成的 MR）
  2. 从算子目录读取 input_specs（可选，无则用 RandomGenerator 默认值）
  3. 调用 RandomGenerator 生成 n_samples 组随机输入
  4. 对每组输入执行原始算子 + MR 变换后算子（dict kwargs 风格，与 MRPreChecker 一致）
  5. 验证 oracle（MRVerifier）
  6. 记录结果到 ResultsManager
  7. 返回可读摘要
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepmt.analysis.reporting.evidence_collector import EvidenceCollector, EvidencePack
from deepmt.analysis.verification.mr_prechecker import MRPreChecker
from deepmt.analysis.verification.mr_verifier import MRVerifier
from deepmt.analysis.verification.random_generator import RandomGenerator
from deepmt.core.logger import logger
from deepmt.core.plugins_manager import FrameworkType, get_plugins_manager
from deepmt.core.results_manager import ResultsManager
from deepmt.ir import MetamorphicRelation, OperatorIR, OracleResult
from deepmt.mr_generator.base.mr_repository import MRRepository
from deepmt.mr_generator.base.operator_catalog import OperatorCatalog


@dataclass
class MRTestSummary:
    """单条 MR 在一个算子上的测试摘要"""


    mr_id: str
    description: str
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
class OperatorTestSummary:
    """单个算子批量测试的汇总结果"""

    operator: str
    framework: str
    mr_count: int
    n_samples: int
    total_cases: int
    passed: int
    failed: int
    errors: int
    mr_summaries: List[MRTestSummary] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)  # 捕获到的证据包 ID 列表

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator,
            "framework": self.framework,
            "mr_count": self.mr_count,
            "n_samples": self.n_samples,
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 4),
            "evidence_ids": self.evidence_ids,
            "mr_summaries": [
                {
                    "mr_id": m.mr_id,
                    "description": m.description,
                    "passed": m.passed,
                    "failed": m.failed,
                    "errors": m.errors,
                    "total": m.total,
                    "pass_rate": round(m.pass_rate, 4),
                }
                for m in self.mr_summaries
            ],
        }


class BatchTestRunner:
    """
    批量测试执行器。

    使用 MR 知识库中已生成的 MR，自动生成随机输入，批量执行蜕变测试。

    与 TestRunner 的区别：
      - TestRunner：依赖 ir_to_code（位置参数风格），输入需从外部传入
      - BatchTestRunner：使用 dict kwargs 风格（与 MRPreChecker 一致），
        自动从算子目录读取 input_specs 并由 RandomGenerator 生成输入

    用法示例：
        runner = BatchTestRunner()
        summary = runner.run_operator("torch.nn.functional.relu", "pytorch", n_samples=10)
        print(f"passed={summary.passed}/{summary.total_cases}")
    """

    def __init__(
        self,
        repo: Optional[MRRepository] = None,
        catalog: Optional[OperatorCatalog] = None,
        results_manager: Optional[ResultsManager] = None,
        evidence_collector: Optional[EvidenceCollector] = None,
        backend_override: Optional[Any] = None,
    ):
        self.repo = repo or MRRepository()
        self.catalog = catalog or OperatorCatalog()
        self.results_manager = results_manager or ResultsManager()
        self.evidence_collector = evidence_collector or EvidenceCollector()
        self.random_gen = RandomGenerator()
        self.verifier = MRVerifier()
        self._backend_override = backend_override  # 用于注入 FaultyPyTorchPlugin 等

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def run_operator(
        self,
        operator_name: str,
        framework: FrameworkType,
        n_samples: int = 10,
        verified_only: bool = False,
        mr_id: Optional[str] = None,
        operator_func: Optional[Any] = None,
        collect_evidence: bool = False,
    ) -> OperatorTestSummary:
        """
        对单个算子执行批量蜕变测试。

        Args:
            operator_name:    算子名称（需与 MR 知识库中的键一致，如 "torch.nn.functional.relu"）
            framework:        目标框架
            n_samples:        每条 MR 的随机测试样本数
            verified_only:    仅使用已验证（verified=True）的 MR
            mr_id:            若指定，则只测试该 MR
            operator_func:    可选，直接传入算子函数（跳过 _resolve_operator）；
                              用于变异测试时注入人工错误实现
            collect_evidence: 失败时是否捕获可复现证据包并保存（默认 False）

        Returns:
            OperatorTestSummary，含逐条 MR 的通过/失败/异常统计
        """
        fw_str = str(framework)
        logger.info(f"[BATCH] Testing {operator_name} on {fw_str} | n_samples={n_samples}")

        # 1. 加载 MR
        mrs = self.repo.load(operator_name, framework=fw_str)
        if verified_only:
            mrs = [m for m in mrs if m.verified]
        if mr_id is not None:
            mrs = [m for m in mrs if m.id == mr_id]

        if not mrs:
            logger.warning(f"[BATCH] No MRs found for {operator_name!r} (framework={fw_str})")
            return OperatorTestSummary(
                operator=operator_name,
                framework=fw_str,
                mr_count=0,
                n_samples=n_samples,
                total_cases=0,
                passed=0,
                failed=0,
                errors=0,
            )

        # 2. 获取框架后端与 input_specs（支持 backend_override 注入缺陷插件）
        try:
            backend = self._backend_override or get_plugins_manager().get_backend(framework)
        except KeyError as e:
            logger.error(f"[BATCH] Plugin not found for {fw_str}: {e}")
            return OperatorTestSummary(
                operator=operator_name,
                framework=fw_str,
                mr_count=len(mrs),
                n_samples=n_samples,
                total_cases=0,
                passed=0,
                failed=0,
                errors=len(mrs) * n_samples,
            )

        input_specs = self._get_input_specs(operator_name, framework)

        # 3. 解析算子函数（若外部传入则直接使用，用于变异测试）
        if operator_func is None:
            try:
                operator_func = backend._resolve_operator(operator_name)
            except ValueError as e:
                logger.error(f"[BATCH] Cannot resolve operator {operator_name!r}: {e}")
                return OperatorTestSummary(
                    operator=operator_name,
                    framework=fw_str,
                    mr_count=len(mrs),
                    n_samples=n_samples,
                    total_cases=0,
                    passed=0,
                    failed=0,
                    errors=len(mrs) * n_samples,
                )

        # 4. 逐 MR 执行
        op_ir = OperatorIR(name=operator_name, input_specs=input_specs)
        all_oracle_results: List[tuple[MetamorphicRelation, OracleResult]] = []
        evidence_packs: List[EvidencePack] = []
        mr_summaries: List[MRTestSummary] = []
        total_passed = total_failed = total_errors = 0

        for mr in mrs:
            mr_summary = self._run_single_mr(
                mr=mr,
                operator_func=operator_func,
                input_specs=input_specs,
                backend=backend,
                n_samples=n_samples,
                oracle_results_out=all_oracle_results,
                evidence_out=evidence_packs if collect_evidence else None,
                operator_name=operator_name,
                framework=fw_str,
            )
            mr_summaries.append(mr_summary)
            total_passed += mr_summary.passed
            total_failed += mr_summary.failed
            total_errors += mr_summary.errors

        # 5. 持久化结果
        if all_oracle_results:
            self.results_manager.store_result(op_ir, all_oracle_results, fw_str)

        # 6. 保存证据包
        saved_ids: List[str] = []
        for pack in evidence_packs:
            self.evidence_collector.save(pack)
            saved_ids.append(pack.evidence_id)

        summary = OperatorTestSummary(
            operator=operator_name,
            framework=fw_str,
            mr_count=len(mrs),
            n_samples=n_samples,
            total_cases=total_passed + total_failed,
            passed=total_passed,
            failed=total_failed,
            errors=total_errors,
            mr_summaries=mr_summaries,
            evidence_ids=saved_ids,
        )
        logger.info(
            f"[BATCH] {operator_name}: "
            f"passed={summary.passed}/{summary.total_cases} "
            f"errors={summary.errors}"
        )
        return summary

    def run_batch(
        self,
        framework: FrameworkType,
        operators: Optional[List[str]] = None,
        category: Optional[str] = None,
        n_samples: int = 10,
        verified_only: bool = False,
    ) -> List[OperatorTestSummary]:
        """
        批量测试多个算子。

        Args:
            framework:     目标框架
            operators:     算子名称列表；为 None 时从 MR 知识库中自动获取所有有 MR 的算子
            category:      算子分类过滤（如 "activation"），None 表示不过滤
            n_samples:     每条 MR 的随机测试样本数
            verified_only: 仅使用已验证的 MR

        Returns:
            每个算子对应一个 OperatorTestSummary 的列表
        """
        if operators is None:
            operators = self.repo.list_operators_by_framework(str(framework))

        if category:
            catalog_names = {e.name for e in self.catalog.get_by_category(framework, category)}
            operators = [op for op in operators if op in catalog_names]

        results = []
        for op in operators:
            result = self.run_operator(
                op,
                framework,
                n_samples=n_samples,
                verified_only=verified_only,
            )
            results.append(result)

        return results

    # ── 私有实现 ──────────────────────────────────────────────────────────────

    def _run_single_mr(
        self,
        mr: MetamorphicRelation,
        operator_func: Any,
        input_specs: List[Dict[str, Any]],
        backend: Any,
        n_samples: int,
        oracle_results_out: List,
        evidence_out: Optional[List[EvidencePack]] = None,
        operator_name: str = "",
        framework: str = "",
    ) -> MRTestSummary:
        """对单条 MR 运行 n_samples 次测试，将 (MR, OracleResult) 追加到 oracle_results_out。

        Args:
            evidence_out: 若不为 None，失败时创建 EvidencePack 并追加到此列表
                          （每条 MR 最多保存 1 个证据包，避免冗余）
        """
        passed = failed = errors = 0
        evidence_captured = False  # 每条 MR 只捕获第一个失败

        # 编译 transform（与 MRPreChecker 共用静态方法）
        bound_transform = MRPreChecker._bind_transform_code(mr.transform_code, operator_func)
        if bound_transform is None:
            logger.warning(f"[BATCH] Cannot bind transform for MR {mr.id[:8]}: {mr.description}")
            errors = n_samples
            return MRTestSummary(
                mr_id=mr.id,
                description=mr.description,
                passed=0,
                failed=0,
                errors=errors,
            )

        for i in range(n_samples):
            try:
                inputs = self.random_gen.generate(input_specs, backend)
                kwargs = MRPreChecker._build_kwargs(inputs)

                orig_output = operator_func(**kwargs)

                transformed_kwargs = bound_transform(kwargs)
                if not isinstance(transformed_kwargs, dict):
                    logger.debug(
                        f"[BATCH] MR {mr.id[:8]} sample {i+1}: "
                        f"transform returned {type(transformed_kwargs)}, expected dict"
                    )
                    errors += 1
                    continue

                trans_output = operator_func(**transformed_kwargs)

                x_input = kwargs.get("input", kwargs.get("x"))
                oracle_result = self.verifier.verify(
                    orig_output, trans_output, mr, backend, x_input=x_input
                )

                oracle_results_out.append((mr, oracle_result))
                if oracle_result.passed:
                    passed += 1
                else:
                    failed += 1
                    logger.debug(
                        f"[BATCH] MR {mr.id[:8]} sample {i+1} FAIL: {oracle_result.detail}"
                    )
                    # 证据捕获：每条 MR 只保留第一个失败样本
                    if evidence_out is not None and not evidence_captured:
                        try:
                            pack = self.evidence_collector.create(
                                operator=operator_name,
                                framework=framework,
                                mr_id=mr.id,
                                mr_description=mr.description,
                                transform_code=mr.transform_code,
                                oracle_expr=mr.oracle_expr,
                                input_tensor=x_input if x_input is not None else inputs[0],
                                actual_diff=oracle_result.actual_diff,
                                tolerance=oracle_result.tolerance,
                                detail=oracle_result.detail,
                            )
                            evidence_out.append(pack)
                            evidence_captured = True
                        except Exception as ev_err:
                            logger.debug(f"[EVIDENCE] Failed to capture: {ev_err}")

            except Exception as e:
                errors += 1
                logger.debug(f"[BATCH] MR {mr.id[:8]} sample {i+1} error: {e}")

        return MRTestSummary(
            mr_id=mr.id,
            description=mr.description,
            passed=passed,
            failed=failed,
            errors=errors,
        )

    def _get_input_specs(
        self, operator_name: str, framework: FrameworkType
    ) -> List[Dict[str, Any]]:
        """从算子目录获取 input_specs；若未找到则返回空列表（RandomGenerator 将使用默认值）。"""
        entry = self.catalog.get_operator_info(framework, operator_name)
        if entry and entry.input_specs:
            return entry.input_specs
        return []
