"""
实验环境信息记录器。

职责：
  在实验开始前快照当前运行环境，确保实验可复现、可审计。
  生成的环境摘要可嵌入 RunManifest 并写入实验记录文件。

环境信息包含：
  - Python 版本
  - 操作系统与平台
  - 各框架已安装版本（pytorch / tensorflow / paddlepaddle / numpy）
  - CPU 核数、内存大小（粗粒度）
  - DeepMT 版本
  - 是否有 GPU 可用（PyTorch 报告）

用法::

    from deepmt.experiments.runs.environment_recorder import EnvironmentRecorder

    env = EnvironmentRecorder().capture()
    print(env.python_version)
    print(env.framework_versions)
    d = env.to_dict()
"""

import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class EnvironmentSnapshot:
    """实验运行时环境快照。"""

    captured_at: str
    """捕获时间（ISO 格式）。"""

    python_version: str
    """Python 版本字符串，如 '3.12.0'。"""

    platform_info: str
    """操作系统与硬件描述，如 'Linux-5.15-x86_64'。"""

    framework_versions: Dict[str, str] = field(default_factory=dict)
    """各框架版本字典 {name: version_or_NOT_INSTALLED}。"""

    cpu_count: Optional[int] = None
    """逻辑 CPU 核数。"""

    total_memory_gb: Optional[float] = None
    """系统总内存（GB，粗粒度，不精确）。"""

    has_gpu: Optional[bool] = None
    """是否检测到可用 GPU（PyTorch CUDA 可用时为 True）。"""

    gpu_info: str = ""
    """GPU 描述字符串（如有）。"""

    deepmt_version: str = ""
    """DeepMT 包版本。"""

    def to_dict(self) -> Dict:
        return {
            "captured_at": self.captured_at,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "framework_versions": dict(self.framework_versions),
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "has_gpu": self.has_gpu,
            "gpu_info": self.gpu_info,
            "deepmt_version": self.deepmt_version,
        }

    def format_text(self) -> str:
        """返回人类可读的环境摘要字符串。"""
        lines = [
            f"环境快照 [{self.captured_at[:19]}]",
            f"  Python:     {self.python_version}",
            f"  Platform:   {self.platform_info}",
            f"  CPU 核数:   {self.cpu_count or 'unknown'}",
            f"  内存(GB):   {self.total_memory_gb or 'unknown'}",
            f"  GPU:        {'是' if self.has_gpu else '否'}"
            + (f" ({self.gpu_info})" if self.gpu_info else ""),
            f"  DeepMT:     {self.deepmt_version or 'unknown'}",
            "  框架版本:",
        ]
        for fw, ver in self.framework_versions.items():
            lines.append(f"    {fw:<16} {ver}")
        return "\n".join(lines)


class EnvironmentRecorder:
    """实验环境信息捕获器。"""

    def capture(self) -> EnvironmentSnapshot:
        """
        快照当前运行环境，返回 EnvironmentSnapshot。

        此方法尽量容错：任何子捕获失败只记录 'unknown' 而不抛出异常。
        """
        snapshot = EnvironmentSnapshot(
            captured_at=datetime.now().isoformat(),
            python_version=platform.python_version(),
            platform_info=self._platform_info(),
        )
        snapshot.cpu_count = self._cpu_count()
        snapshot.total_memory_gb = self._memory_gb()
        snapshot.framework_versions = self._framework_versions()
        snapshot.has_gpu, snapshot.gpu_info = self._gpu_info()
        snapshot.deepmt_version = self._deepmt_version()
        return snapshot

    # ── 内部捕获方法 ──────────────────────────────────────────────────────────

    def _platform_info(self) -> str:
        try:
            return f"{platform.system()}-{platform.release()}-{platform.machine()}"
        except Exception:
            return "unknown"

    def _cpu_count(self) -> Optional[int]:
        try:
            import os
            return os.cpu_count()
        except Exception:
            return None

    def _memory_gb(self) -> Optional[float]:
        try:
            import os
            # psutil 可选；若无则尝试 /proc/meminfo（Linux）
            try:
                import psutil
                return round(psutil.virtual_memory().total / (1024 ** 3), 1)
            except ImportError:
                pass
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return round(kb / (1024 ** 2), 1)
        except Exception:
            pass
        return None

    def _framework_versions(self) -> Dict[str, str]:
        from deepmt.experiments.version_matrix import get_installed_versions
        return get_installed_versions()

    def _gpu_info(self):
        """返回 (has_gpu: bool, gpu_info_str: str)。"""
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                names = [torch.cuda.get_device_name(i) for i in range(count)]
                return True, f"{count}x [{', '.join(names)}]"
            return False, ""
        except Exception:
            return None, ""

    def _deepmt_version(self) -> str:
        try:
            from importlib.metadata import version
            return version("deepmt")
        except Exception:
            pass
        try:
            import deepmt
            return getattr(deepmt, "__version__", "")
        except Exception:
            return ""
