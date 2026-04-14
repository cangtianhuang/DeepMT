"""应用层知识检索与上下文构建器（J2）。

不依赖实时网络搜索；以预定义的领域事实 + 场景元数据构建结构化 LLM 提示上下文。
未来若需集成 web search，可在 build() 中注入搜索结果。

输出的上下文字符串直接插入 ApplicationLLMMRGenerator 的 user prompt。
"""

from deepmt.mr_generator.application.scenario import ApplicationScenario


class AppContextBuilder:
    """应用层上下文构建器。

    将 ApplicationScenario 的元数据与领域知识组织为 LLM 可直接使用的结构化文本。

    用法::

        builder = AppContextBuilder()
        context = builder.build(scenario)
    """

    def build(self, scenario: ApplicationScenario) -> str:
        """构建用于 LLM 提示的应用层上下文字符串。

        Args:
            scenario: 应用场景描述对象

        Returns:
            结构化上下文字符串（Markdown 格式）
        """
        lines = [
            f"## Application Scenario: {scenario.name}",
            "",
            f"- **Task Type**: {scenario.task_type}",
            f"- **Domain**: {scenario.domain}",
            f"- **Description**: {scenario.description}",
            "",
            f"### Input Format",
            f"{scenario.input_schema}",
            "",
            f"### Output Format",
            f"{scenario.output_schema}",
            "",
        ]

        if scenario.domain_facts:
            lines.append("### Domain Knowledge")
            for i, fact in enumerate(scenario.domain_facts, 1):
                lines.append(f"{i}. {fact}")
            lines.append("")

        if scenario.sample_inputs:
            lines.append("### Sample Input Format")
            lines.append(f"Input dict example: `{scenario.sample_inputs[0]}`")
            lines.append(f"Expected label example: `{scenario.sample_labels[0] if scenario.sample_labels else 'N/A'}`")
            lines.append("")

        return "\n".join(lines)

    def build_source_snapshot(self, scenario: ApplicationScenario) -> dict:
        """返回上下文来源快照，便于追踪和人工复核。

        Returns:
            含场景名称、领域事实数量、样例数等信息的字典
        """
        return {
            "scenario_name": scenario.name,
            "task_type": scenario.task_type,
            "domain": scenario.domain,
            "num_domain_facts": len(scenario.domain_facts),
            "num_sample_inputs": len(scenario.sample_inputs),
            "source_type": "static_knowledge",  # 静态知识（非实时搜索）
        }
