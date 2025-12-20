"""
算子层LLM MR生成器：使用大语言模型生成算子MR猜想（路径A）
专为算子层MR生成设计，因为大模型MR生成方法在后续也可能用于其他层次
"""

import json
import uuid
from typing import List, Optional, Dict, Any

from ir.schema import MetamorphicRelation
from tools.llm.client import LLMClient
from core.logger import get_logger


class OperatorLLMMRGenerator:
    """算子层LLM MR生成器：使用LLM生成算子MR猜想"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        model: str = None,
    ):
        """
        初始化算子层LLM MR生成器

        Args:
            llm_client: LLM客户端（如果为None则创建）
            api_key: API密钥（如果为None，则从配置文件或环境变量获取）
            model: 使用的模型名称（如果为None，则从配置文件获取）
        """
        self.logger = get_logger()

        if llm_client:
            self.llm_client = llm_client
            self.llm_available = True
        else:
            try:
                from tools.llm.client import LLMClient

                self.llm_client = LLMClient(api_key=api_key, model=model)
                self.llm_available = True
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM client: {e}")

    def _build_prompt(
        self,
        operator_name: str,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        num_inputs: int = 2,
    ) -> str:
        """
        构建LLM提示

        Args:
            operator_name: 算子名称
            operator_code: 算子代码（可选）
            operator_doc: 算子文档（可选）
            num_inputs: 输入数量

        Returns:
            提示字符串
        """
        prompt = f"""你是一个蜕变测试专家。请为以下深度学习算子生成蜕变关系（MR）。

算子信息：
- 名称：{operator_name}
- 输入数量：{num_inputs}
"""

        if operator_code:
            prompt += f"- 代码：\n```python\n{operator_code}\n```\n"

        if operator_doc:
            prompt += f"- 文档：\n{operator_doc}\n"

        prompt += """
要求：
1. 生成Top-5个高质量的MR猜想
2. 每个MR应该描述输入变换和期望的输出关系
3. 使用清晰的数学或自然语言描述
4. 考虑常见的数学性质：交换律、结合律、分配律、单位元、零元等

输出格式（必须是有效的JSON）：
{
    "mrs": [
        {
            "description": "MR的数学描述，如：f(x, y) == f(y, x)",
            "input_transform": "输入变换的描述，如：交换x和y",
            "expected_relation": "期望关系类型：equal, proportional, invariant等",
            "transform_code": "Python lambda表达式，如：lambda x, y: (y, x)"
        }
    ]
}

只返回JSON，不要包含其他文字说明。
"""
        return prompt

    def generate_mr_candidates(
        self,
        operator_name: str,
        operator_code: Optional[str] = None,
        operator_doc: Optional[str] = None,
        num_inputs: int = 2,
        top_k: int = 5,
    ) -> List[MetamorphicRelation]:
        """
        使用LLM生成MR候选列表（路径A）

        Args:
            operator_name: 算子名称
            operator_code: 算子代码（可选）
            operator_doc: 算子文档（可选）
            num_inputs: 输入数量
            top_k: 生成Top-K个MR（默认5）

        Returns:
            MR候选列表
        """
        if not self.llm_available or not self.llm_client:
            self.logger.warning("LLM not available, returning empty list")
            return []

        try:
            # 构建提示
            prompt = self._build_prompt(
                operator_name, operator_code, operator_doc, num_inputs
            )

            # 调用LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in metamorphic testing for deep learning operators.",
                },
                {"role": "user", "content": prompt},
            ]

            content = self.llm_client.chat_completion(
                messages, temperature=0.7, max_tokens=1000
            )

            # 提取JSON（可能包含markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 解析JSON
            data = json.loads(content)

            # 转换为MR对象
            mrs = []
            for mr_data in data.get("mrs", [])[:top_k]:
                try:
                    mr = self._parse_mr_response(mr_data, num_inputs)
                    if mr:
                        mrs.append(mr)
                except Exception as e:
                    self.logger.warning(f"Failed to parse MR: {e}")
                    continue

            self.logger.info(
                f"Generated {len(mrs)} MR candidates from LLM for {operator_name}"
            )
            return mrs

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response content: {content[:500]}")
            return []
        except Exception as e:
            self.logger.error(f"LLM MR generation error: {e}")
            return []

    def _parse_mr_response(
        self, mr_data: Dict[str, Any], num_inputs: int
    ) -> Optional[MetamorphicRelation]:
        """
        解析LLM响应的MR数据

        Args:
            mr_data: MR数据字典
            num_inputs: 输入数量

        Returns:
            MetamorphicRelation对象，如果解析失败则返回None
        """
        try:
            description = mr_data.get("description", "")
            expected = mr_data.get("expected_relation", "equal")

            # 解析transform_code
            transform_code = mr_data.get("transform_code", "")
            if transform_code:
                # 执行lambda表达式
                transform = eval(transform_code)
            else:
                # 如果没有提供代码，尝试从描述推断
                transform = self._infer_transform_from_description(
                    mr_data.get("input_transform", ""), num_inputs
                )

            return MetamorphicRelation(
                id=str(uuid.uuid4()),
                description=description,
                transform=transform,
                expected=expected,
                tolerance=1e-6,
                layer="operator",
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse MR data: {e}")
            return None

    def _infer_transform_from_description(
        self, description: str, num_inputs: int
    ) -> callable:
        """
        从描述推断变换函数（fallback方法）

        Args:
            description: 输入变换描述
            num_inputs: 输入数量

        Returns:
            变换函数
        """
        description_lower = description.lower()

        # 简单的启发式规则
        if (
            "交换" in description
            or "swap" in description_lower
            or "commut" in description_lower
        ):
            if num_inputs >= 2:
                return lambda *args: (args[1], args[0]) + args[2:]

        # 默认：恒等变换
        return lambda *args: args
