"""模型层 MR 生成器：将结构分析结果与变换策略结合，生成模型层 MetamorphicRelation。

生成流程：
  1. 接受 ModelIR（含 model_instance）或 ModelAnalysisResult
  2. 调用 GraphAnalyzer 提取结构摘要（若未分析）
  3. 由 TransformStrategyLibrary 选择适用策略
  4. 将每条策略转换为 MetamorphicRelation 记录
  5. 返回 MR 列表（可选：存入 MRRepository）

生成的 MR 使用：
  - layer = "model"
  - subject_type = "model"
  - transform_code = strategy.transform_code（lambda x: ...）
  - oracle_expr = strategy.oracle_type[:oracle_params] 编码字符串

oracle_expr 编码规则（ModelVerifier 解析）：
  - "prediction_consistent"        → argmax(orig) == argmax(trans)（按样本比较）
  - "topk_consistent:3"            → top-3 预测集合相同
  - "output_close:1e-5"            → allclose(orig, trans, atol=1e-5)
  - "output_order_invariant"       → 翻转 batch 后 orig.flip(0) ≈ trans（按行比较）
"""

import uuid
from typing import List, Optional

from deepmt.core.logger import logger
from deepmt.ir.schema import MetamorphicRelation, ModelIR
from deepmt.model.graph_analyzer import ModelAnalysisResult, ModelGraphAnalyzer
from deepmt.mr_generator.model.transform_strategy import (
    TransformStrategy,
    TransformStrategyLibrary,
)


def _encode_oracle_expr(strategy: TransformStrategy) -> str:
    """将策略的 oracle_type + oracle_params 编码为 oracle_expr 字符串。"""
    otype = strategy.oracle_type
    params = strategy.oracle_params

    if otype == "prediction_consistent":
        return "prediction_consistent"
    if otype == "topk_consistent":
        k = params.get("k", 3)
        return f"topk_consistent:{k}"
    if otype == "output_close":
        atol = params.get("atol", 1e-5)
        return f"output_close:{atol}"
    if otype == "output_order_invariant":
        return "output_order_invariant"
    # 默认：直接返回 oracle_type
    return otype


class ModelMRGenerator:
    """模型层 MR 生成器。

    基于结构分析结果和变换策略模板生成 MetamorphicRelation 列表。
    不依赖 LLM，纯模板驱动（Phase I 第一版）。

    用法::

        gen = ModelMRGenerator()
        mrs = gen.generate(model_ir)
        for mr in mrs:
            print(mr.id, mr.description, mr.oracle_expr)
    """

    def __init__(
        self,
        strategy_library: Optional[TransformStrategyLibrary] = None,
        analyzer: Optional[ModelGraphAnalyzer] = None,
    ):
        self.strategy_library = strategy_library or TransformStrategyLibrary()
        self.analyzer = analyzer or ModelGraphAnalyzer()

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def generate(
        self,
        model_ir: ModelIR,
        max_per_model: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """为给定模型生成 MR 列表。

        Args:
            model_ir:      已填充 model_instance 的 ModelIR 对象
            max_per_model: 最多生成的 MR 数量；None 表示生成所有适用 MR

        Returns:
            MetamorphicRelation 列表（layer="model", subject_type="model"）
        """
        # 1. 结构分析
        analysis = self._get_analysis(model_ir)
        if analysis is None:
            return []

        # 2. 选择策略
        strategies = self.strategy_library.select(analysis, max_strategies=max_per_model)
        logger.info(
            f"[ModelMRGenerator] {model_ir.name}: "
            f"type={analysis.model_type}, selected {len(strategies)} strategies"
        )

        # 3. 生成 MR
        mrs = []
        for strategy in strategies:
            mr = self._strategy_to_mr(strategy, model_ir, analysis)
            mrs.append(mr)

        logger.info(
            f"[ModelMRGenerator] Generated {len(mrs)} MRs for {model_ir.name}"
        )
        return mrs

    def generate_from_analysis(
        self,
        analysis: ModelAnalysisResult,
        model_name: str,
        framework: str = "pytorch",
        max_per_model: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """直接从 ModelAnalysisResult 生成 MR（不需要 model_instance）。

        Args:
            analysis:      ModelAnalysisResult 对象
            model_name:    模型名称（用于 MR 的 subject_name）
            framework:     框架名称
            max_per_model: 最多生成的 MR 数量

        Returns:
            MetamorphicRelation 列表
        """
        strategies = self.strategy_library.select(analysis, max_strategies=max_per_model)
        dummy_ir = ModelIR(
            name=model_name,
            framework=framework,
            model_type=analysis.model_type,
            task_type=analysis.task_type,
            input_shape=analysis.input_shape,
            output_shape=analysis.output_shape,
            num_classes=analysis.num_classes,
        )
        return [self._strategy_to_mr(s, dummy_ir, analysis) for s in strategies]

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def _get_analysis(self, model_ir: ModelIR) -> Optional[ModelAnalysisResult]:
        """获取或执行结构分析。"""
        if model_ir.model_instance is None:
            logger.error(
                f"[ModelMRGenerator] model_instance 为 None，"
                f"请先通过 ModelBenchmarkRegistry.get(with_instance=True) 获取 {model_ir.name}"
            )
            return None
        try:
            return self.analyzer.analyze(model_ir)
        except Exception as e:
            logger.error(f"[ModelMRGenerator] 结构分析失败 ({model_ir.name}): {e}")
            return None

    def _strategy_to_mr(
        self,
        strategy: TransformStrategy,
        model_ir: ModelIR,
        analysis: ModelAnalysisResult,
    ) -> MetamorphicRelation:
        """将策略转换为 MetamorphicRelation。"""
        oracle_expr = _encode_oracle_expr(strategy)
        mr_id = str(uuid.uuid4())
        description = (
            f"[{analysis.model_type.upper()}] {strategy.description}"
            if strategy.description
            else f"[{analysis.model_type.upper()}] {strategy.name}"
        )
        return MetamorphicRelation(
            id=mr_id,
            description=description,
            subject_name=model_ir.name,
            subject_type="model",
            transform_code=strategy.transform_code,
            oracle_expr=oracle_expr,
            category=strategy.category,
            layer="model",
            source="template",
            applicable_frameworks=[model_ir.framework] if model_ir.framework else None,
            lifecycle_state="pending",
            checked=None,
            proven=None,
            verified=False,
        )
