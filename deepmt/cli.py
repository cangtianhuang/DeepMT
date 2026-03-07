"""
DeepMT 顶层 CLI 入口

使用方式:
    python -m deepmt <command> [options]

命令组:
    mr       MR 生成与管理
    test     测试执行
    repo     MR 知识库管理
    catalog  算子目录浏览与查询
    data     数据目录管理
    health   系统健康检查
"""

import sys
from pathlib import Path

import click

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from deepmt.commands.mr import mr
from deepmt.commands.test import test
from deepmt.commands.repo import repo
from deepmt.commands.health import health
from deepmt.commands.catalog import catalog
from deepmt.commands.data import data

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option("0.1.0", "-V", "--version", prog_name="deepmt")
def cli():
    """DeepMT — 深度学习框架蜕变关系自动生成与分层测试系统。

    \b
    命令组:
      mr       MR 生成与管理（算子层）
      test     测试执行
      repo     MR 知识库管理
      catalog  算子目录浏览与跨框架查询
      data     数据目录管理（日志清理等）
      health   系统健康检查

    \b
    快速开始:
      deepmt catalog list --framework pytorch       # 列出 PyTorch 所有算子
      deepmt catalog search relu                    # 跨框架搜索 relu
      deepmt catalog info relu                      # 查询 relu 的框架分布和 MR 数量
      deepmt mr generate relu --save                # 为 relu 生成并保存 MR
      deepmt test operator relu                     # 运行蜕变测试
      deepmt data clean-logs --keep-days 7          # 清理 7 天前的日志
    """


cli.add_command(mr)
cli.add_command(test)
cli.add_command(repo)
cli.add_command(health)
cli.add_command(catalog)
cli.add_command(data)
