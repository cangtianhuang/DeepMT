"""
MR 模板池：从配置文件读取算子无关的通用数学律模板，用于 MR 候选生成。

设计原则：
  - 模板字段只存放**算子无关的通用数学律**（交换律、结合律、线性性、对称性等）。
  - 不存放"算子→模板"映射，也不存放针对特定算子的预写答案。
  - 运行期一律通过 `discover_all_templates` 按算子 arity 枚举候选，
    是否真正成立交由下游 precheck + SymPy 证明判定。
"""

import inspect
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from deepmt.core.logger import logger
from deepmt.ir import MetamorphicRelation


@dataclass
class MRTemplate:
    """MR模板数据结构"""

    name: str  # 模板名称
    description: str  # MR描述
    transform_code: str  # 原始 transform_code 字符串（来自 YAML）
    oracle_expr: str  # 框架无关的验证表达式
    category: str = "general"  # MR类别
    min_inputs: int = 1  # 最小输入数量
    max_inputs: Optional[int] = None  # 最大输入数量（None表示无限制）
    tolerance: Optional[float] = None  # 数值容差（None表示使用系统默认值 1e-6）
    property_tags: List[str] = None  # 数学属性标签（commutative / linear / monotone 等）

    def __post_init__(self):
        if self.property_tags is None:
            self.property_tags = []


class MRTemplatePool:
    """MR 模板池：加载算子无关的通用数学律，供候选生成枚举。"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认路径：项目内随代码分发的模板池（git 追踪）
            self.config_path = Path(__file__).parents[1] / "config" / "templates.yaml"
        else:
            self.config_path = Path(config_path)

        self.templates: Dict[str, MRTemplate] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag → [template_name, ...]

        self._load_config()

    def _load_config(self):
        """从配置文件加载模板（只解析 YAML，不 eval transform_code）"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            for template_name, tdata in config.get("templates", {}).items():
                try:
                    tags = tdata.get("property_tags") or []
                    tmpl = MRTemplate(
                        name=tdata.get("name", template_name),
                        description=tdata.get("description", ""),
                        transform_code=tdata.get("transform_code", ""),
                        oracle_expr=tdata.get("oracle_expr", ""),
                        category=tdata.get("category", "general"),
                        min_inputs=tdata.get("min_inputs", 1),
                        max_inputs=tdata.get("max_inputs"),
                        tolerance=tdata.get("tolerance"),
                        property_tags=list(tags),
                    )
                    self.templates[template_name] = tmpl
                    for tag in tags:
                        self._tag_index.setdefault(tag, []).append(template_name)
                except Exception as e:
                    logger.warning(f"Failed to load template {template_name}: {e}")

            logger.debug(
                f"Loaded {len(self.templates)} MR templates "
                f"({len(self._tag_index)} property tags) from {self.config_path}"
            )

        except FileNotFoundError:
            logger.error(f"MR templates config file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load MR templates config: {e}")
            raise

    def get_templates_by_tag(self, tag: str) -> List[MRTemplate]:
        """按属性标签查询模板。"""
        names = self._tag_index.get(tag, [])
        return [self.templates[n] for n in names if n in self.templates]

    def available_tags(self) -> List[str]:
        """返回所有已知属性标签（按名称排序）。"""
        return sorted(self._tag_index.keys())

    def _infer_num_inputs(self, operator_func: Optional[Callable]) -> int:
        """推断算子的张量输入数量（仅计无默认值的必填参数）。

        例：relu(input, inplace=False) → 1（inplace 有默认值，不计）。
        """
        if operator_func is None:
            return 2

        try:
            sig = inspect.signature(operator_func)
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            return max(len(required), 1)
        except Exception as e:
            logger.warning(
                f"Failed to infer num_inputs from operator_func: {e}, using default: 2"
            )
            return 2

    def discover_all_templates(
        self,
        operator_func: Optional[Callable] = None,
        num_inputs: Optional[int] = None,
    ) -> List[MRTemplate]:
        """
        枚举全部模板，按算子 arity 过滤。

        这是从算子到模板的**唯一**入口：不依赖任何"算子→模板"先验映射，
        所有筛选交给下游 precheck + SymPy 证明。

        Args:
            operator_func: 算子函数对象（用于推断输入数量）
            num_inputs:    输入数量（优先于 operator_func 推断）

        Returns:
            满足输入数量约束的全部模板列表
        """
        if num_inputs is None:
            num_inputs = self._infer_num_inputs(operator_func)

        result = [
            t for t in self.templates.values()
            if num_inputs >= t.min_inputs
            and (t.max_inputs is None or num_inputs <= t.max_inputs)
        ]
        logger.debug(
            f"[DISCOVER] {len(result)}/{len(self.templates)} templates "
            f"compatible with num_inputs={num_inputs}"
        )
        return result

    def create_mr_from_template(self, template: MRTemplate) -> MetamorphicRelation:
        """从模板创建 MR 对象"""
        transform_code = template.transform_code
        raw_func = eval(transform_code) if transform_code else lambda *args: args

        def transform(*args):
            try:
                return raw_func(*args)
            except TypeError:
                raise  # 让 SymPy prover 的 Path 1 感知到失败，转入 Path 2（dict 协议）
            except Exception as e:
                logger.warning(f"Transform function error: {e}")
                return args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description=template.description,
            transform=transform,
            transform_code=transform_code,
            oracle_expr=template.oracle_expr,
            category=template.category,
            tolerance=template.tolerance if template.tolerance is not None else 1e-6,
            layer="operator",
            source="template",
            verified=False,
        )

    def generate_mr_candidates(
        self,
        operator_name: str,
        operator_func: Optional[Callable] = None,
        num_inputs: Optional[int] = None,
    ) -> List[MetamorphicRelation]:
        """
        为算子枚举模板池候选 MR。

        流程：按算子 arity 过滤出兼容模板，逐一实例化为 MetamorphicRelation。
        是否真正适用于该算子由 precheck / SymPy 判定。
        """
        templates = self.discover_all_templates(
            operator_func=operator_func, num_inputs=num_inputs
        )

        candidates = []
        for template in templates:
            try:
                mr = self.create_mr_from_template(template)
                candidates.append(mr)
            except Exception as e:
                logger.warning(
                    f"Failed to create MR from template {template.name}: {e}"
                )

        logger.info(
            f"⚡ [GEN] Generated {len(candidates)} MR candidates from templates "
            f"for operator {operator_name}"
        )

        return candidates
