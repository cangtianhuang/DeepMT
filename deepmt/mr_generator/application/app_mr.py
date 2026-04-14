"""应用层 MR 生成器（J3+J4）：知识上下文 + LLM 生成 + 模板回退。

生成流程：
  1. 接受 ApplicationIR 或 ApplicationScenario
  2. 调用 AppContextBuilder 构建结构化上下文
  3. 若 use_llm=True，调用 ApplicationLLMMRGenerator 生成候选 MR
     否则，从场景的 known_transforms 生成模板 MR（无 LLM 依赖）
  4. 对候选去重、规范化字段，返回 MetamorphicRelation 列表

生成的 MR 使用：
  - layer = "application"
  - subject_type = "application"
  - oracle_expr = "label_consistent" | "label_consistent_soft:{k}" | "confidence_acceptable:{d}"
  - source = "llm" | "template"

oracle_expr 含义（SemanticMRValidator 解析）：
  - "label_consistent"              原始与变换输入的预测标签相同
  - "label_consistent_soft:3"       原始标签出现在变换输入的 top-3 预测中
  - "confidence_acceptable:0.1"     置信度下降不超过 0.1
"""

import uuid
from typing import List, Optional, Union

from deepmt.mr_generator.application.scenario import ApplicationScenario
from deepmt.core.logger import logger
from deepmt.ir import ApplicationIR, MetamorphicRelation
from deepmt.mr_generator.application.app_context_builder import AppContextBuilder
from deepmt.mr_generator.application.app_llm_mr_generator import (
    ApplicationLLMMRGenerator,
    _safe_eval_lambda,
)


class ApplicationMRGenerator:
    """应用层 MR 生成器。

    支持两种生成模式：
      - use_llm=True（默认）：调用 LLM 生成语义 MR 候选
      - use_llm=False：使用场景中的 known_transforms 模板（不依赖 LLM/网络）

    用法::

        gen = ApplicationMRGenerator(use_llm=False)
        mrs = gen.generate_from_scenario("ImageClassification")

        gen = ApplicationMRGenerator(use_llm=True)
        mrs = gen.generate_from_ir(app_ir)
    """

    def __init__(
        self,
        use_llm: bool = True,
        context_builder: Optional[AppContextBuilder] = None,
        llm_generator: Optional[ApplicationLLMMRGenerator] = None,
        registry=None,
    ) -> None:
        self.use_llm = use_llm
        self.context_builder = context_builder or AppContextBuilder()
        self._llm_generator: Optional[ApplicationLLMMRGenerator] = llm_generator
        if registry is None:
            from deepmt.benchmarks.applications.app_registry import ApplicationBenchmarkRegistry
            registry = ApplicationBenchmarkRegistry()
        self.registry = registry

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def generate_from_scenario(
        self,
        scenario_name: str,
        max_mrs: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """按场景名称生成 MR 列表。

        Args:
            scenario_name: 场景名称（如 "ImageClassification"、"TextSentiment"）
            max_mrs:       最多生成 MR 数量；None 表示不限制

        Returns:
            MetamorphicRelation 列表
        """
        scenario = self.registry.get(scenario_name)
        if scenario is None:
            logger.error(f"[AppMRGen] 未找到场景: {scenario_name!r}")
            return []
        return self._generate_from_scenario_obj(scenario, max_mrs)

    def generate_from_ir(
        self,
        app_ir: ApplicationIR,
        max_mrs: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """从 ApplicationIR 对象生成 MR 列表。

        若 app_ir.name 对应已注册场景，直接复用场景元数据；
        否则构造临时 ApplicationScenario 并生成。

        Args:
            app_ir:  ApplicationIR 对象
            max_mrs: 最多生成 MR 数量

        Returns:
            MetamorphicRelation 列表
        """
        scenario = self.registry.get(app_ir.name)
        if scenario is not None:
            return self._generate_from_scenario_obj(scenario, max_mrs)

        # 未注册场景：构造临时 ApplicationScenario
        temp_scenario = ApplicationScenario(
            name=app_ir.name,
            task_type=app_ir.task_type,
            domain=app_ir.domain,
            input_type=_infer_input_type(app_ir.task_type),
            output_type=_infer_output_type(app_ir.task_type),
            input_schema=app_ir.input_description,
            output_schema=app_ir.output_description,
            description=f"{app_ir.task_type} application: {app_ir.name}",
            domain_facts=list(app_ir.context_snippets),
            sample_inputs=list(app_ir.sample_inputs),
            sample_labels=list(app_ir.sample_labels),
        )
        return self._generate_from_scenario_obj(temp_scenario, max_mrs)

    def generate(
        self,
        target: Union[str, ApplicationIR, ApplicationScenario],
        max_mrs: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """统一入口：按名称字符串、ApplicationIR 或 ApplicationScenario 生成 MR。"""
        if isinstance(target, str):
            return self.generate_from_scenario(target, max_mrs)
        if isinstance(target, ApplicationIR):
            return self.generate_from_ir(target, max_mrs)
        if isinstance(target, ApplicationScenario):
            return self._generate_from_scenario_obj(target, max_mrs)
        raise TypeError(f"不支持的 target 类型: {type(target)}")

    # ── 内部 ──────────────────────────────────────────────────────────────────

    def _generate_from_scenario_obj(
        self,
        scenario: ApplicationScenario,
        max_mrs: Optional[int],
    ) -> List[MetamorphicRelation]:
        """从 ApplicationScenario 对象生成 MR 列表（内部方法）。"""
        if self.use_llm:
            mrs = self._generate_llm(scenario, max_mrs)
        else:
            mrs = self._generate_template(scenario, max_mrs)

        # 去重（相同 transform_code + oracle_expr）
        mrs = _deduplicate(mrs)

        # 设置 subject_name（以防 LLM 没有填充）
        for mr in mrs:
            if not mr.subject_name:
                mr.subject_name = scenario.name

        if max_mrs is not None:
            mrs = mrs[:max_mrs]

        logger.info(
            f"[AppMRGen] {scenario.name}: 生成 {len(mrs)} 条应用层 MR "
            f"({'llm' if self.use_llm else 'template'})"
        )
        return mrs

    def _generate_llm(
        self, scenario: ApplicationScenario, max_mrs: Optional[int]
    ) -> List[MetamorphicRelation]:
        """调用 LLM 生成候选 MR。"""
        if self._llm_generator is None:
            self._llm_generator = ApplicationLLMMRGenerator()

        context = self.context_builder.build(scenario)
        top_k = max_mrs if max_mrs is not None else 5
        return self._llm_generator.generate_mr_candidates(scenario, context, top_k=top_k)

    def _generate_template(
        self, scenario: ApplicationScenario, max_mrs: Optional[int]
    ) -> List[MetamorphicRelation]:
        """从场景的 known_transforms 生成模板 MR（不依赖 LLM）。"""
        mrs: List[MetamorphicRelation] = []
        transforms = scenario.known_transforms
        if max_mrs is not None:
            transforms = transforms[:max_mrs]

        for t in transforms:
            transform_code = t.get("transform_code", "")
            transform = _safe_eval_lambda(transform_code)
            if transform is None:
                logger.warning(
                    f"[AppMRGen] 模板 transform_code eval 失败，跳过: {transform_code[:60]}"
                )
                continue

            mr = MetamorphicRelation(
                id=str(uuid.uuid4()),
                description=t.get("description", t.get("name", "unknown")),
                subject_name=scenario.name,
                subject_type="application",
                transform_code=transform_code,
                transform=transform,
                oracle_expr=t.get("oracle_expr", "label_consistent"),
                category=t.get("category", "general"),
                layer="application",
                source="template",
                lifecycle_state="pending",
                checked=None,
                proven=None,
                verified=False,
                analysis=(
                    f"[risk:{t.get('risk_level', 'low')}] "
                    f"{t.get('rationale', '')}"
                ),
            )
            mrs.append(mr)

        return mrs


# ── 工具函数 ──────────────────────────────────────────────────────────────────


def _infer_input_type(task_type: str) -> str:
    """根据 task_type 推断 input_type。"""
    if "image" in task_type:
        return "image_array"
    if "text" in task_type or "sentiment" in task_type or "nlp" in task_type:
        return "text_string"
    return "dict"


def _infer_output_type(task_type: str) -> str:
    """根据 task_type 推断 output_type。"""
    if "classification" in task_type:
        return "class_label"
    if "sentiment" in task_type:
        return "sentiment_label"
    return "label"


def _deduplicate(mrs: List[MetamorphicRelation]) -> List[MetamorphicRelation]:
    """去除具有相同 transform_code + oracle_expr 的重复 MR。"""
    seen = set()
    result = []
    for mr in mrs:
        key = (mr.transform_code.strip(), mr.oracle_expr.strip())
        if key not in seen:
            seen.add(key)
            result.append(mr)
    return result
