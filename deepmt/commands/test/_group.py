"""
deepmt.commands.test._group — Click 命令组定义与共享工具

本模块定义 `test` 命令组对象和各子模块共享的常量与辅助函数。
"""

import click

from deepmt._utils import not_implemented_error

_SUPPORTED_FRAMEWORKS = {"pytorch", "paddlepaddle"}
_ALL_FRAMEWORKS = {"pytorch", "tensorflow", "paddlepaddle"}
_MUTANT_TYPES = ["negate", "add_const", "scale", "identity", "zero"]


def _check_framework(framework: str):
    if framework not in _SUPPORTED_FRAMEWORKS:
        not_implemented_error(
            f"--framework {framework}",
            f"框架 '{framework}' 的插件尚未实现，目前仅支持 pytorch。",
        )


@click.group()
def test():
    """测试执行（运行算子 / 从配置文件 / 查看历史）。"""
