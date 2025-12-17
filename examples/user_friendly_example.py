"""
用户友好API示例
演示优化后的使用方式：IR隐藏，MR生成与测试分离
"""

from api.deepmt import DeepMT
from mr_generator.operator_mr import OperatorMRGenerator
from mr_generator.knowledge_base import KnowledgeBase
from mr_generator.mr_repository import MRRepository
from ir.converter import IRConverter
from core.test_runner import TestRunner
from core.plugins_manager import PluginsManager
from core.results_manager import ResultsManager


def example_simple_usage():
    """示例1：最简单的使用方式（IR完全隐藏）"""
    print("=" * 60)
    print("示例1：最简单的使用方式")
    print("=" * 60)

    # 用户只需要3行代码
    client = DeepMT()
    result = client.test_operator("Add", [1.0, 2.0], "pytorch")
    print(result.summary())


def example_batch_testing():
    """示例2：批量测试多个算子"""
    print("\n" + "=" * 60)
    print("示例2：批量测试多个算子")
    print("=" * 60)

    client = DeepMT()

    operators = [
        {"name": "Add", "inputs": [1.0, 2.0]},
        {"name": "Multiply", "inputs": [3.0, 4.0]},
        {"name": "Subtract", "inputs": [10.0, 3.0]},
    ]

    results = client.test_operators(operators, "pytorch")

    print("\n批量测试结果：")
    for result in results:
        print(result.summary())


def example_mr_generation_separation():
    """示例3：MR生成与测试分离"""
    print("\n" + "=" * 60)
    print("示例3：MR生成与测试分离")
    print("=" * 60)

    # 步骤1：生成MR并保存到知识库
    print("\n[步骤1] 生成MR并保存到知识库...")

    ir_converter = IRConverter()
    add_ir = ir_converter.from_operator_name("Add", [1.0, 2.0])

    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    mrs = generator.generate(add_ir)

    mr_repo = MRRepository()
    saved_count = mr_repo.save("Add", mrs)
    print(f"    已保存 {saved_count} 个MR到知识库")

    # 步骤2：从知识库加载MR并测试
    print("\n[步骤2] 从知识库加载MR并测试...")

    loaded_mrs = mr_repo.load("Add")
    print(f"    从知识库加载了 {len(loaded_mrs)} 个MR")

    # 使用TestRunner执行测试（不生成MR）
    plugins = PluginsManager()
    plugins.load_plugins()
    results_manager = ResultsManager()

    test_runner = TestRunner(plugins, results_manager)
    test_runner.run_with_mrs(add_ir, loaded_mrs, "pytorch")

    print("    测试完成！")


def example_mr_reuse():
    """示例4：MR重用 - 同一个MR测试多个框架"""
    print("\n" + "=" * 60)
    print("示例4：MR重用 - 同一个MR测试多个框架")
    print("=" * 60)

    # 生成一次MR
    print("\n[1] 生成MR...")
    ir_converter = IRConverter()
    add_ir = ir_converter.from_operator_name("Add", [1.0, 2.0])

    kb = KnowledgeBase()
    generator = OperatorMRGenerator(kb)
    mrs = generator.generate(add_ir)

    mr_repo = MRRepository()
    mr_repo.save("Add", mrs)
    print(f"    生成了 {len(mrs)} 个MR并保存")

    # 使用同一个MR测试不同框架（如果有多个插件）
    print("\n[2] 使用同一个MR测试不同框架...")

    plugins = PluginsManager()
    plugins.load_plugins()
    results_manager = ResultsManager()
    test_runner = TestRunner(plugins, results_manager)

    # 加载MR
    loaded_mrs = mr_repo.load("Add")

    # 测试PyTorch（如果可用）
    if "pytorch" in plugins.list_plugins():
        print("    测试 PyTorch...")
        test_runner.run_with_mrs(add_ir, loaded_mrs, "pytorch")

    # 可以继续测试其他框架
    # if "tensorflow" in plugins.list_plugins():
    #     print("    测试 TensorFlow...")
    #     test_runner.run_with_mrs(add_ir, loaded_mrs, "tensorflow")


def example_config_file():
    """示例5：使用配置文件"""
    print("\n" + "=" * 60)
    print("示例5：使用配置文件")
    print("=" * 60)

    # 创建配置文件
    import yaml

    config = {
        "tests": [
            {
                "type": "operator",
                "name": "Add",
                "inputs": [1.0, 2.0],
                "framework": "pytorch",
            },
            {
                "type": "operator",
                "name": "Multiply",
                "inputs": [3.0, 4.0],
                "framework": "pytorch",
            },
        ]
    }

    config_path = "tests/config_example.yaml"
    from pathlib import Path

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print(f"    配置文件已创建: {config_path}")

    # 从配置文件运行测试
    client = DeepMT()
    results = client.test_from_config(config_path)

    print("\n    测试结果：")
    for result in results:
        print(result.summary())


def main():
    """运行所有示例"""
    try:
        example_simple_usage()
        example_batch_testing()
        example_mr_generation_separation()
        example_mr_reuse()
        example_config_file()

        print("\n" + "=" * 60)
        print("所有示例完成！")
        print("=" * 60)
        print("\n优化总结：")
        print("  1. ✅ IR对用户完全隐藏")
        print("  2. ✅ MR生成与测试分离")
        print("  3. ✅ MR可以持久化和重用")
        print("  4. ✅ 用户只需简单的API调用")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
