"""
deepmt.commands.test — 测试执行子命令组

命令:
    operator      测试单个算子
    batch         批量测试（从 MR 知识库自动选取算子，RandomGenerator 生成输入）
    model         模型层蜕变测试（基于结构分析，无需 LLM）
    mutate        变异测试（注入已知错误实现，验证缺陷检测能力）
    open          开放测试（对含预设缺陷的插件运行批量测试，模拟真实框架缺陷场景）
    report        生成测试结果报告
    dedup         缺陷线索去重（将失败证据包聚类为独立缺陷线索）
    evidence      证据包管理（list / show / script）
    cross         跨框架一致性测试（对比两个框架在等价算子上的行为）
    from-config   从 YAML 配置文件批量测试
    history       查看测试历史
    failures      查看失败的测试用例
"""

# 1. 先导入 test 命令组对象（无循环依赖）
from deepmt.commands.test._group import test  # noqa: F401

# 2. 导入各子模块触发 @test.command() 装饰器注册（顺序无关）
from deepmt.commands.test import execution  # noqa: F401
from deepmt.commands.test import analysis   # noqa: F401
from deepmt.commands.test import evidence   # noqa: F401
from deepmt.commands.test import history    # noqa: F401

__all__ = ["test"]
