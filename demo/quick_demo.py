"""
快速演示：算子层MR测试闭环
"""

from ir.schema import OperatorIR
from mr_generator.operator.operator_mr import OperatorMRGenerator
from mr_generator.operator.knowledge_base import KnowledgeBase
from core.scheduler import TaskScheduler
from core.ir_manager import IRManager
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager


def main():
    """运行快速演示"""
    print("=" * 60)
    print("DeepMT 快速演示：算子层MR测试")
    print("=" * 60)

    # 1. 定义算子IR（可以手动构建，也可以从文件加载）
    print("\n[1] 创建算子IR...")

    # 方式1：手动构建IR（当前方式）
    add_ir = OperatorIR(
        name="Add", inputs=[1.0, 2.0], outputs=[], properties={"commutative": True}
    )
    print(f"   算子: {add_ir.name}, 输入: {add_ir.inputs}")

    # （可选）方式2：从文件加载IR
    # ir_manager = IRManager()
    # add_ir = ir_manager.load_ir("tests/sample_ops.json")

    # （可选）保存IR以便后续使用
    # ir_manager.save_ir(add_ir, "tests/add_operator.json")

    # 2. 初始化MR生成器
    # 注意：MRGenerator不是用来生成IR的，而是基于IR生成MR（蜕变关系）！
    print("\n[2] 初始化MR生成器...")
    print("    说明：MRGenerator将基于IR生成蜕变关系（如交换律、结合律等）")
    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    print("   MR生成器已初始化")

    # 演示：基于IR生成MR
    print("\n[2.1] 基于IR生成MR...")
    mrs = generator.generate(add_ir)
    print(f"    生成了 {len(mrs)} 个MR:")
    for i, mr in enumerate(mrs, 1):
        print(f"      MR {i}: {mr.description}")

    # 3. 初始化插件管理器
    print("\n[3] 加载框架插件...")
    plugins = PluginsManager()
    plugins.load_plugins()
    print(f"   已加载插件: {', '.join(plugins.list_plugins())}")

    # 4. 初始化结果管理器
    print("\n[4] 初始化结果管理器...")
    results_manager = ResultsManager()
    print("   结果管理器已初始化")

    # 5. 初始化IR管理器
    print("\n[5] 初始化IR管理器...")
    ir_manager = IRManager()
    print("   IR管理器已初始化")

    # 6. 创建任务调度器并运行
    print("\n[6] 创建任务调度器并运行测试...")
    scheduler = TaskScheduler(
        ir_manager=ir_manager,
        mr_generator=generator,
        plugins_manager=plugins,
        results_manager=results_manager,
    )

    # 运行测试任务
    scheduler.run_task(add_ir, "pytorch")

    # 7. 查看结果摘要
    print("\n[7] 查看测试结果摘要...")
    summary = results_manager.get_summary(add_ir.name)
    if summary:
        for item in summary:
            print(f"   算子: {item['ir_name']}")
            print(f"   总测试数: {item['total_tests']}")
            print(f"   通过: {item['passed_tests']}")
            print(f"   失败: {item['failed_tests']}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
