"""
IR持久化示例
演示如何保存和加载IR对象
"""

from pathlib import Path
from ir.schema import OperatorIR, ModelIR
from core.ir_manager import IRManager


def example_save_and_load_operator_ir():
    """示例：保存和加载算子IR"""
    print("=" * 60)
    print("示例1：算子IR的保存和加载")
    print("=" * 60)

    ir_manager = IRManager()

    # 1. 创建算子IR
    print("\n[1] 创建算子IR...")
    add_ir = OperatorIR(
        name="Add",
        inputs=[1.0, 2.0],
        outputs=[],
        properties={"commutative": True, "associative": True},
    )
    print(f"    算子: {add_ir.name}")
    print(f"    输入: {add_ir.inputs}")

    # 2. 保存IR为JSON
    print("\n[2] 保存IR为JSON...")
    json_path = Path("tests/add_operator.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    ir_manager.save_ir(add_ir, json_path, format="json")
    print(f"    已保存到: {json_path}")

    # 3. 保存IR为YAML
    print("\n[3] 保存IR为YAML...")
    yaml_path = Path("tests/add_operator.yaml")
    ir_manager.save_ir(add_ir, yaml_path, format="yaml")
    print(f"    已保存到: {yaml_path}")

    # 4. 从JSON加载IR
    print("\n[4] 从JSON加载IR...")
    loaded_ir = ir_manager.load_ir(json_path)
    print(f"    加载的算子: {loaded_ir.name}")
    print(f"    输入: {loaded_ir.inputs}")
    print(f"    属性: {loaded_ir.properties}")

    # 5. 验证加载的IR
    print("\n[5] 验证加载的IR...")
    is_valid = ir_manager.validate_ir(loaded_ir)
    print(f"    IR有效性: {is_valid}")


def example_batch_operators():
    """示例：批量保存多个算子IR"""
    print("\n" + "=" * 60)
    print("示例2：批量保存多个算子IR")
    print("=" * 60)

    ir_manager = IRManager()
    operators = [
        OperatorIR(
            name="Add", inputs=[1.0, 2.0], outputs=[], properties={"commutative": True}
        ),
        OperatorIR(
            name="Multiply",
            inputs=[3.0, 4.0],
            outputs=[],
            properties={"commutative": True},
        ),
        OperatorIR(
            name="Subtract",
            inputs=[10.0, 3.0],
            outputs=[],
            properties={"anti_commutative": True},
        ),
    ]

    print("\n批量保存算子IR...")
    for op in operators:
        path = Path(f"tests/{op.name.lower()}_operator.json")
        ir_manager.save_ir(op, path)
        print(f"  已保存: {op.name} -> {path}")

    print("\n批量加载算子IR...")
    for op in operators:
        path = Path(f"tests/{op.name.lower()}_operator.json")
        if path.exists():
            loaded = ir_manager.load_ir(path)
            print(f"  已加载: {loaded.name}")


def example_model_ir():
    """示例：模型IR的保存和加载"""
    print("\n" + "=" * 60)
    print("示例3：模型IR的保存和加载")
    print("=" * 60)

    ir_manager = IRManager()

    # 创建模型IR
    print("\n[1] 创建模型IR...")
    model_ir = ModelIR(
        name="SimpleCNN",
        layers=[
            {"type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64}},
            {"type": "ReLU", "params": {}},
            {"type": "MaxPool2d", "params": {"kernel_size": 2}},
        ],
        connections=[
            (0, 1),
            (1, 2),
        ],
    )
    print(f"    模型: {model_ir.name}")
    print(f"    层数: {len(model_ir.layers)}")

    # 保存模型IR
    print("\n[2] 保存模型IR...")
    path = Path("tests/simple_cnn_model.json")
    ir_manager.save_ir(model_ir, path)
    print(f"    已保存到: {path}")

    # 加载模型IR
    print("\n[3] 加载模型IR...")
    loaded_model = ir_manager.load_ir(path)
    print(f"    加载的模型: {loaded_model.name}")
    print(f"    层数: {len(loaded_model.layers)}")


def main():
    """运行所有示例"""
    try:
        example_save_and_load_operator_ir()
        example_batch_operators()
        example_model_ir()

        print("\n" + "=" * 60)
        print("所有示例完成！")
        print("=" * 60)
        print("\n提示：")
        print("  - IR可以保存为JSON或YAML格式")
        print("  - 保存的IR可以用于后续测试和复现")
        print("  - 建议将常用IR保存到tests/目录下")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
