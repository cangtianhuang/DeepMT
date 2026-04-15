"""
Phase O · O7 — 插件契约合规性反射测试

对 deepmt.plugins.PLUGIN_REGISTRY 中每个 is_available() 为真的插件：
  - 校验类继承 FrameworkPlugin
  - 校验必需原语都已实现（未通过 ABC 的 abstractmethod 保留默认值）
  - 校验 framework_name / framework_version / supported_operators 返回值类型
"""

import pytest

from deepmt.plugins import PLUGIN_REGISTRY
from deepmt.plugins.framework_plugin import FrameworkPlugin

REQUIRED_PRIMITIVES = [
    "_to_tensor",
    "_execute_operator",
    "to_numpy",
    "get_shape",
    "make_tensor",
    "allclose",
    "eval_expr",
    "element_compare",
]


@pytest.mark.parametrize("entry", PLUGIN_REGISTRY, ids=lambda e: e.name)
def test_registry_entry_loadable(entry):
    cls = entry.load_class()
    assert issubclass(cls, FrameworkPlugin), f"{cls.__name__} 必须继承 FrameworkPlugin"
    assert isinstance(cls.framework_name(), str) and cls.framework_name()
    assert isinstance(cls.framework_version(), str)
    assert isinstance(cls.supported_operators(), list)


@pytest.mark.parametrize("entry", PLUGIN_REGISTRY, ids=lambda e: e.name)
def test_primitives_not_abstract_when_available(entry):
    if not entry.is_available():
        pytest.skip(f"{entry.name} 运行时不可用，跳过原语实例化")
    cls = entry.load_class()
    # 实例化不应抛异常（说明 _require_* 通过）
    plugin = cls()
    for name in REQUIRED_PRIMITIVES:
        fn = getattr(plugin, name, None)
        assert callable(fn), f"{cls.__name__}.{name} 未实现"


def test_pytorch_numpy_paddle_all_report_concrete_versions():
    """硬依赖 / 已安装可选依赖必须给出非 unknown / non-uninstalled 版本号。"""
    for entry in PLUGIN_REGISTRY:
        if not entry.is_available():
            continue
        version = entry.load_class().framework_version()
        assert version not in ("unknown", "uninstalled", "error"), \
            f"{entry.name} 报告版本 {version!r}，应返回真实运行时版本"
