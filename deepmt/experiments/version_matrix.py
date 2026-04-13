"""
论文实验框架版本矩阵。

固化论文实验所用的框架版本，确保实验可复现、可引用。
实际运行时自动检测已安装版本，并与固定版本对比，提示不一致。

用法::

    from deepmt.experiments.version_matrix import VERSION_MATRIX, get_installed_versions

    for entry in VERSION_MATRIX:
        print(entry.framework, entry.pinned_version, entry.role)

    installed = get_installed_versions()
    print(installed)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FrameworkVersionEntry:
    """单个框架的版本矩阵条目。"""

    framework: str
    """框架标识（与 FrameworkType 一致）。"""

    pinned_version: str
    """论文实验固定版本（如 '2.1.0'）。"""

    role: str
    """在实验中的角色：primary / secondary / reference。"""

    python_min: str = "3.10"
    """最低 Python 版本要求。"""

    notes: str = ""
    """额外说明（安装方式、已知问题等）。"""


# ── 固定版本矩阵 ──────────────────────────────────────────────────────────────
#
# 修改本矩阵需同步更新 docs/dev/status.md 和 requirements.txt。

VERSION_MATRIX: List[FrameworkVersionEntry] = [
    FrameworkVersionEntry(
        framework="pytorch",
        pinned_version="2.1.0",
        role="primary",
        notes="论文主框架。CPU 版本用于 CI；GPU 版本可选。",
    ),
    FrameworkVersionEntry(
        framework="tensorflow",
        pinned_version="2.14.0",
        role="secondary",
        notes="跨框架对比框架（Phase H+）。接口占位实现。",
    ),
    FrameworkVersionEntry(
        framework="paddlepaddle",
        pinned_version="2.5.2",
        role="reference",
        notes="第三框架占位，当前仅 numpy 后端运行。",
    ),
    FrameworkVersionEntry(
        framework="numpy",
        pinned_version="1.26.0",
        role="reference",
        notes="算子层部分 MR 的基准参考实现后端。",
    ),
]

# 按 framework name 索引
_MATRIX_BY_FRAMEWORK: Dict[str, FrameworkVersionEntry] = {
    e.framework: e for e in VERSION_MATRIX
}


def get_entry(framework: str) -> Optional[FrameworkVersionEntry]:
    """获取指定框架的版本矩阵条目，未找到返回 None。"""
    return _MATRIX_BY_FRAMEWORK.get(framework)


def get_installed_versions() -> Dict[str, str]:
    """
    检测当前环境中各框架的已安装版本。

    Returns:
        {framework_name: installed_version_or_"NOT_INSTALLED"}
    """
    result: Dict[str, str] = {}

    # PyTorch
    try:
        import torch
        result["pytorch"] = torch.__version__
    except ImportError:
        result["pytorch"] = "NOT_INSTALLED"

    # TensorFlow
    try:
        import tensorflow as tf
        result["tensorflow"] = tf.__version__
    except ImportError:
        result["tensorflow"] = "NOT_INSTALLED"

    # PaddlePaddle
    try:
        import paddle
        result["paddlepaddle"] = paddle.__version__
    except ImportError:
        result["paddlepaddle"] = "NOT_INSTALLED"

    # NumPy
    try:
        import numpy as np
        result["numpy"] = np.__version__
    except ImportError:
        result["numpy"] = "NOT_INSTALLED"

    return result


def check_version_compatibility() -> List[Dict[str, str]]:
    """
    对比固定版本与已安装版本，返回不一致条目列表。

    Returns:
        List of {framework, pinned, installed, status}
        status 取值：ok / mismatch / not_installed
    """
    installed = get_installed_versions()
    report = []
    for entry in VERSION_MATRIX:
        actual = installed.get(entry.framework, "NOT_INSTALLED")
        if actual == "NOT_INSTALLED":
            status = "not_installed"
        elif actual.startswith(entry.pinned_version):
            status = "ok"
        else:
            status = "mismatch"
        report.append(
            {
                "framework": entry.framework,
                "pinned": entry.pinned_version,
                "installed": actual,
                "status": status,
                "role": entry.role,
            }
        )
    return report
