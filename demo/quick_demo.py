# quick_demo.py
from ir.schema import OperatorIR
from mr_generator.operator_mr import OperatorMRGenerator
from core.scheduler import TaskScheduler
from core.ir_manager import IRManager
from core.plugins_manager import PluginsManager

# 定义算子IR
add_ir = OperatorIR(
    name="Add", inputs=[1, 2], outputs=[], properties={"commutative": True}
)

# MR生成
kb = {"Add": ["f(x,y) == f(y,x)"]}
generator = OperatorMRGenerator(kb)

# 插件
plugins = PluginsManager()
plugins.load_plugins()

# 调度执行
scheduler = TaskScheduler(IRManager(), generator, plugins, None)
scheduler.run_task(add_ir, "pytorch")
