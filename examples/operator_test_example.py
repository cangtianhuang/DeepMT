"""
算子层MR测试示例
演示如何使用DeepMT测试不同算子的蜕变关系
"""
from ir.schema import OperatorIR
from mr_generator.operator_mr import OperatorMRGenerator
from mr_generator.knowledge_base import KnowledgeBase
from core.scheduler import TaskScheduler
from core.ir_manager import IRManager
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager


def test_add_operator():
    """测试加法算子的MR"""
    print("\n" + "="*60)
    print("测试：Add 算子")
    print("="*60)
    
    # 创建算子IR
    add_ir = OperatorIR(
        name="Add",
        inputs=[3.0, 5.0],
        outputs=[],
        properties={"commutative": True, "associative": True}
    )
    
    # 初始化组件
    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    plugins = PluginsManager()
    plugins.load_plugins()
    results_manager = ResultsManager()
    ir_manager = IRManager()
    
    # 创建调度器并运行
    scheduler = TaskScheduler(
        ir_manager=ir_manager,
        mr_generator=generator,
        plugins_manager=plugins,
        results_manager=results_manager
    )
    
    scheduler.run_task(add_ir, "pytorch")
    
    # 查看结果
    summary = results_manager.get_summary(add_ir.name)
    if summary:
        print(f"\n测试结果：")
        for item in summary:
            print(f"  总测试数: {item['total_tests']}")
            print(f"  通过: {item['passed_tests']}")
            print(f"  失败: {item['failed_tests']}")


def test_multiply_operator():
    """测试乘法算子的MR"""
    print("\n" + "="*60)
    print("测试：Multiply 算子")
    print("="*60)
    
    multiply_ir = OperatorIR(
        name="Multiply",
        inputs=[2.0, 4.0],
        outputs=[],
        properties={"commutative": True, "associative": True}
    )
    
    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    plugins = PluginsManager()
    plugins.load_plugins()
    results_manager = ResultsManager()
    ir_manager = IRManager()
    
    scheduler = TaskScheduler(
        ir_manager=ir_manager,
        mr_generator=generator,
        plugins_manager=plugins,
        results_manager=results_manager
    )
    
    scheduler.run_task(multiply_ir, "pytorch")
    
    summary = results_manager.get_summary(multiply_ir.name)
    if summary:
        print(f"\n测试结果：")
        for item in summary:
            print(f"  总测试数: {item['total_tests']}")
            print(f"  通过: {item['passed_tests']}")
            print(f"  失败: {item['failed_tests']}")


def test_subtract_operator():
    """测试减法算子的MR"""
    print("\n" + "="*60)
    print("测试：Subtract 算子")
    print("="*60)
    
    subtract_ir = OperatorIR(
        name="Subtract",
        inputs=[10.0, 3.0],
        outputs=[],
        properties={"anti_commutative": True}
    )
    
    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    plugins = PluginsManager()
    plugins.load_plugins()
    results_manager = ResultsManager()
    ir_manager = IRManager()
    
    scheduler = TaskScheduler(
        ir_manager=ir_manager,
        mr_generator=generator,
        plugins_manager=plugins,
        results_manager=results_manager
    )
    
    scheduler.run_task(subtract_ir, "pytorch")
    
    summary = results_manager.get_summary(subtract_ir.name)
    if summary:
        print(f"\n测试结果：")
        for item in summary:
            print(f"  总测试数: {item['total_tests']}")
            print(f"  通过: {item['passed_tests']}")
            print(f"  失败: {item['failed_tests']}")


def main():
    """运行所有算子测试"""
    print("\n" + "="*60)
    print("DeepMT 算子层MR测试套件")
    print("="*60)
    
    try:
        test_add_operator()
        test_multiply_operator()
        test_subtract_operator()
        
        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


