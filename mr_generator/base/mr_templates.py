"""
MR模板池：从配置文件读取常见数学变换模板
用于路径B（无知识）的MR猜想生成
"""

import uuid
import yaml
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass

from ir.schema import MetamorphicRelation
from core.logger import get_logger


@dataclass
class MRTemplate:
    """MR模板数据结构"""

    name: str  # 模板名称
    description: str  # MR描述
    transform_func: Callable  # 输入变换函数
    expected: str  # 期望关系类型
    min_inputs: int = 1  # 最小输入数量
    max_inputs: Optional[int] = None  # 最大输入数量（None表示无限制）


class MRTemplatePool:
    """MR模板池：从配置文件读取和管理常见数学变换模板"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模板池

        Args:
            config_path: 配置文件路径（如果为None，则使用默认路径）
        """
        self.logger = get_logger()
        self.templates: Dict[str, MRTemplate] = {}
        self.operator_mr_mapping: Dict[str, List[str]] = {}

        if config_path is None:
            # 默认配置文件路径：mr_generator/config/mr_templates.yaml
            base_dir = Path(__file__).parent.parent
            default_path = base_dir / "config" / "mr_templates.yaml"
            self.config_path = default_path
        else:
            self.config_path = Path(config_path)

        self._load_config()

    def _load_config(self):
        """从配置文件加载模板"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 加载算子到MR的映射
            self.operator_mr_mapping = config.get("operator_mr_mapping", {})

            # 加载模板定义
            templates_config = config.get("templates", {})

            for template_name, template_data in templates_config.items():
                try:
                    # 解析transform_code为函数
                    transform_code = template_data.get("transform_code", "")
                    transform_func = (
                        eval(transform_code) if transform_code else lambda *args: args
                    )

                    template = MRTemplate(
                        name=template_data.get("name", template_name),
                        description=template_data.get("description", ""),
                        transform_func=transform_func,
                        expected=template_data.get("expected", "equal"),
                        min_inputs=template_data.get("min_inputs", 1),
                        max_inputs=template_data.get("max_inputs"),
                    )

                    self.templates[template_name] = template
                except Exception as e:
                    self.logger.warning(f"Failed to load template {template_name}: {e}")

            self.logger.info(
                f"Loaded {len(self.templates)} MR templates from {self.config_path}"
            )

        except FileNotFoundError:
            self.logger.error(f"MR templates config file not found: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load MR templates config: {e}")
            raise

    def get_applicable_templates(
        self, operator_name: str, num_inputs: int
    ) -> List[MRTemplate]:
        """
        获取适用于指定算子的模板列表

        Args:
            operator_name: 算子名称
            num_inputs: 输入数量

        Returns:
            适用的模板列表
        """
        applicable = []

        # 从算子到MR的映射中获取适用的模板名称
        template_names = self.operator_mr_mapping.get(operator_name, [])

        for template_name in template_names:
            template = self.templates.get(template_name)
            if template is None:
                continue

            # 检查输入数量
            if num_inputs < template.min_inputs:
                continue
            if template.max_inputs is not None and num_inputs > template.max_inputs:
                continue

            applicable.append(template)

        # 如果没有找到特定映射，尝试通用模板（identity）
        if not applicable and "identity" in self.templates:
            identity_template = self.templates["identity"]
            if num_inputs >= identity_template.min_inputs:
                if (
                    identity_template.max_inputs is None
                    or num_inputs <= identity_template.max_inputs
                ):
                    applicable.append(identity_template)

        self.logger.debug(
            f"Found {len(applicable)} applicable templates for {operator_name} "
            f"(inputs: {num_inputs})"
        )

        return applicable

    def create_mr_from_template(
        self, template: MRTemplate, operator_inputs: List[Any]
    ) -> MetamorphicRelation:
        """
        从模板创建MR对象

        Args:
            template: MR模板
            operator_inputs: 算子输入

        Returns:
            MetamorphicRelation对象
        """

        # 创建变换函数
        def transform(*args):
            try:
                return template.transform_func(*args)
            except Exception as e:
                self.logger.warning(f"Transform function error: {e}")
                return args

        return MetamorphicRelation(
            id=str(uuid.uuid4()),
            description=template.description,
            transform=transform,
            expected=template.expected,
            tolerance=1e-6,
            layer="operator",
        )

    def generate_mr_candidates(
        self, operator_name: str, operator_inputs: List[Any]
    ) -> List[MetamorphicRelation]:
        """
        为算子生成MR候选列表（路径B：模板池）

        Args:
            operator_name: 算子名称
            operator_inputs: 算子输入

        Returns:
            MR候选列表
        """
        num_inputs = len(operator_inputs)
        templates = self.get_applicable_templates(operator_name, num_inputs)

        candidates = []
        for template in templates:
            try:
                mr = self.create_mr_from_template(template, operator_inputs)
                candidates.append(mr)
            except Exception as e:
                self.logger.warning(
                    f"Failed to create MR from template {template.name}: {e}"
                )

        self.logger.info(
            f"Generated {len(candidates)} MR candidates from templates "
            f"for operator {operator_name}"
        )

        return candidates
