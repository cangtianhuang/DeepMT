"""
含预设缺陷的 PyTorch 插件（用于 D5 开放测试受控实验）。

设计目的：
  模拟真实深度学习框架中的潜在缺陷，让 DeepMT 能在"受控的真实场景"中测试缺陷检测能力，
  而不依赖于真实 PyTorch 版本中的已知 bug。

缺陷注入机制：
  FaultyPyTorchPlugin 继承 PyTorchPlugin，覆盖 _resolve_operator()：
  在返回算子函数之前，若该算子出现在激活的缺陷表中，则用 create_mutant_func 包装。

激活方式（优先级从高到低）：
  1. 构造参数 fault_specs: Dict[str, str]，直接传入（代码级控制，适合测试）
  2. 环境变量 DEEPMT_INJECT_FAULTS，格式：
       "all"                          — 激活内置缺陷目录中的所有算子
       "torch.relu:negate,torch.exp:add_const"  — 显式指定算子:变异类型对

内置缺陷目录（BUILTIN_FAULT_CATALOG）：
  每个条目描述一种真实框架中可能出现的微妙缺陷，故意设计为不会在普通测试中被发现，
  但蜕变关系能够检测到。

环境变量：
  DEEPMT_INJECT_FAULTS  — 缺陷注入规格（见上），不设置则不注入任何缺陷
"""

import os
from typing import Callable, Dict, Optional, Tuple

from deepmt.core.logger import logger
from deepmt.plugins.pytorch_plugin import PyTorchPlugin

# ── 缺陷注入环境变量 ──────────────────────────────────────────────────────────

DEEPMT_INJECT_FAULTS_ENV = "DEEPMT_INJECT_FAULTS"

# ── 内置缺陷目录 ──────────────────────────────────────────────────────────────
# 格式：operator_name -> (mutant_type_str, kwargs, description)
# 每个条目对应一种真实框架中可能出现的微妙 bug
#
# 设计原则：
#   - 缺陷要"微妙"：普通数值测试难以发现，但 MR 能暴露
#   - 缺陷要"有物理意义"：对应真实框架代码中可能出现的错误

BUILTIN_FAULT_CATALOG: Dict[str, Tuple[str, Dict, str]] = {
    # relu: 负值截断阈值错误（允许极小负值通过）
    "torch.nn.functional.relu": (
        "add_const",
        {"const": -1e-3},
        "relu 负值截断阈值偏移：实现中 clamp(min=-1e-3) 而非 clamp(min=0)，"
        "小负值能够通过，破坏非负性",
    ),
    # exp: 乘以错误系数（数值精度问题或错误的 fast-math 近似）
    "torch.exp": (
        "scale",
        {"scale": 1.001},
        "exp 输出乘以 1.001（模拟 fast-math 近似误差），破坏 exp(a+b)==exp(a)*exp(b)",
    ),
    # log: 加了常数偏置（防止 log(0) 的 epsilon 实现错误）
    "torch.log": (
        "add_const",
        {"const": 1e-4},
        "log 输出加入常数偏置 1e-4（epsilon 保护实现错误），破坏 log(a*b)==log(a)+log(b)",
    ),
    # sigmoid: 输出取反符号（激活函数正负极性实现错误）
    "torch.nn.functional.sigmoid": (
        "negate",
        {},
        "sigmoid 输出取反（正负号错误），所有输出均违反 [0,1] 范围约束",
    ),
    # softmax: 恒等映射（softmax 权重归一化未实现）
    "torch.nn.functional.softmax": (
        "identity",
        {},
        "softmax 返回原始输入（未执行归一化），破坏输出和为1的约束",
    ),
    # abs: 错误缩放（绝对值实现缺少精度调整）
    "torch.abs": (
        "scale",
        {"scale": 0.999},
        "abs 输出乘以 0.999（模拟量化误差），破坏 abs(-x)==abs(x)",
    ),
    # tanh: 输出加偏置（激活函数常数项错误）
    "torch.tanh": (
        "add_const",
        {"const": 0.01},
        "tanh 输出加 0.01（常数项 bug），破坏 tanh 奇函数性质 tanh(-x)==-tanh(x)",
    ),
    # sqrt: 错误缩放（平方根实现的数值近似误差）
    "torch.sqrt": (
        "scale",
        {"scale": 1.002},
        "sqrt 输出乘以 1.002（Newton 迭代终止过早），破坏 sqrt(a)^2==a",
    ),
    # cos: 输出乘以 1.05（模拟 SIMD 精度近似错误），破坏 |cos(x)|<=1 有界性与周期性
    "torch.cos": (
        "scale",
        {"scale": 1.05},
        "cos 输出乘以 1.05（SIMD 精度近似错误），破坏 |cos(x)|<=1 有界性，"
        "且破坏 cos(x+2π)==cos(x) 周期性（两侧按同比例放大但与 oracle 不合）",
    ),
    # cosh: 恒等映射（严重破坏偶函数性质：f(-x)!=f(x) 对非零输入）
    "torch.cosh": (
        "identity",
        {},
        "cosh 返回原始输入（近似分支彻底失效），破坏 cosh(-x)==cosh(x) 偶函数性质",
    ),
    # asinh: 输出取反（破坏 asinh 全局单调递增性质）
    "torch.asinh": (
        "negate",
        {},
        "asinh 输出取反（符号错误），破坏 asinh 单调递增 x+1 >= x ⇒ f(x+1) >= f(x)",
    ),
    # expm1: 输出取反（破坏 expm1 单调递增性质）
    "torch.expm1": (
        "negate",
        {},
        "expm1 输出取反（符号错误），破坏 expm1 单调递增 x+1 >= x ⇒ f(x+1) >= f(x)",
    ),
}


# ── FaultyPyTorchPlugin ───────────────────────────────────────────────────────


class FaultyPyTorchPlugin(PyTorchPlugin):
    """
    含预设缺陷的 PyTorch 插件。

    通过覆盖 _resolve_operator() 将真实算子替换为注入了已知缺陷的包装函数。
    缺陷配置由构造参数或环境变量控制，其余行为与 PyTorchPlugin 完全一致。

    Args:
        fault_specs: 算子 → 变异类型的映射，如 {"torch.relu": "negate"}。
                     None 则从 DEEPMT_INJECT_FAULTS 环境变量读取。
                     空字典 {} 表示不注入任何缺陷（等价于正常插件）。

    用法示例（代码）：
        plugin = FaultyPyTorchPlugin(fault_specs={
            "torch.nn.functional.relu": "add_const",
            "torch.exp": "scale",
        })
        relu = plugin._resolve_operator("torch.nn.functional.relu")
        # relu 现在是有缺陷的版本

    用法示例（环境变量）：
        DEEPMT_INJECT_FAULTS=all  deepmt test open --operator torch.exp
        DEEPMT_INJECT_FAULTS=torch.exp:scale,torch.relu:negate  deepmt test open
    """

    def __init__(
        self,
        fault_specs: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        # fault_specs 优先于环境变量
        if fault_specs is not None:
            self._active_faults = self._resolve_specs(fault_specs)
        else:
            self._active_faults = self._parse_env()

        if self._active_faults:
            logger.info(
                f"[FAULTY] Injecting faults into {len(self._active_faults)} operator(s): "
                f"{list(self._active_faults.keys())}"
            )
        else:
            logger.debug("[FAULTY] No faults active (FaultyPyTorchPlugin behaves like PyTorchPlugin)")

    # ── 核心覆盖 ──────────────────────────────────────────────────────────────

    def _resolve_operator(self, name: str) -> Callable:
        real_func = super()._resolve_operator(name)
        if name not in self._active_faults:
            return real_func

        mutant_type_str, kwargs, description = self._active_faults[name]
        logger.debug(f"[FAULTY] {name}: {description}")

        from deepmt.analysis.reporting.mutation_tester import MutantType, create_mutant_func
        mutant_type = MutantType(mutant_type_str)
        return create_mutant_func(real_func, mutant_type, **kwargs)

    # ── 缺陷目录查询 ─────────────────────────────────────────────────────────

    @staticmethod
    def list_catalog() -> Dict[str, Tuple[str, str]]:
        """返回内置缺陷目录 {operator: (mutant_type, description)}，用于 CLI 展示。"""
        return {
            op: (mt, desc)
            for op, (mt, _, desc) in BUILTIN_FAULT_CATALOG.items()
        }

    @staticmethod
    def active_faults_from_env() -> Dict[str, str]:
        """解析环境变量，返回 {operator: mutant_type}（不含 kwargs）。"""
        raw = os.environ.get(DEEPMT_INJECT_FAULTS_ENV, "").strip()
        if not raw:
            return {}
        if raw.lower() == "all":
            return {op: mt for op, (mt, _, _) in BUILTIN_FAULT_CATALOG.items()}
        result = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                op, mt = part.rsplit(":", 1)
                result[op.strip()] = mt.strip()
            else:
                # 仅指定算子名，使用内置缺陷目录的默认变异类型
                if part in BUILTIN_FAULT_CATALOG:
                    result[part] = BUILTIN_FAULT_CATALOG[part][0]
                else:
                    logger.warning(f"[FAULTY] Unknown operator in DEEPMT_INJECT_FAULTS: {part!r}")
        return result

    # ── 内部辅助 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_env() -> Dict[str, Tuple[str, Dict, str]]:
        """从环境变量解析并填充完整的 (mutant_type, kwargs, description) 三元组。"""
        raw = os.environ.get(DEEPMT_INJECT_FAULTS_ENV, "").strip()
        if not raw:
            return {}
        if raw.lower() == "all":
            return dict(BUILTIN_FAULT_CATALOG)

        result = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                op, mt = part.rsplit(":", 1)
                op, mt = op.strip(), mt.strip()
            else:
                op, mt = part.strip(), None

            if op in BUILTIN_FAULT_CATALOG:
                catalog_mt, catalog_kwargs, catalog_desc = BUILTIN_FAULT_CATALOG[op]
                effective_mt = mt or catalog_mt
                result[op] = (effective_mt, catalog_kwargs, catalog_desc)
            elif mt:
                # 算子不在目录但显式指定了变异类型，使用空 kwargs
                result[op] = (mt, {}, f"custom fault: {mt} on {op}")
            else:
                logger.warning(f"[FAULTY] {op!r} not in BUILTIN_FAULT_CATALOG, skipping")

        return result

    @staticmethod
    def _resolve_specs(specs: Dict[str, str]) -> Dict[str, Tuple[str, Dict, str]]:
        """将用户传入的 {op: mutant_type} 映射扩充为完整三元组，尽量从目录获取 kwargs。"""
        result = {}
        for op, mt in specs.items():
            if op in BUILTIN_FAULT_CATALOG and BUILTIN_FAULT_CATALOG[op][0] == mt:
                result[op] = BUILTIN_FAULT_CATALOG[op]
            else:
                # 从目录获取 kwargs（若算子在目录中），否则使用空 kwargs
                kwargs = BUILTIN_FAULT_CATALOG[op][1] if op in BUILTIN_FAULT_CATALOG else {}
                result[op] = (mt, kwargs, f"injected fault: {mt} on {op}")
        return result
