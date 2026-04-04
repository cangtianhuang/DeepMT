"""
快速演示：通过 DeepMT 高层 API 测试算子
"""

from deepmt import DeepMT


def main():
    print("=" * 60)
    print("DeepMT 快速演示：算子层 MR 测试")
    print("=" * 60)

    client = DeepMT()

    # 单算子测试
    print("\n[1] 测试单个算子 Add ...")
    result = client.test_operator("Add", [1.0, 2.0], framework="pytorch")
    print(result.summary())

    # 批量测试
    print("\n[2] 批量测试多个算子 ...")
    results = client.test_operators(
        [
            {"name": "Add", "inputs": [1.0, 2.0]},
            {"name": "Multiply", "inputs": [3.0, 4.0]},
        ],
        framework="pytorch",
    )
    for r in results:
        print(r.summary())

    # 查看历史
    print("\n[3] 查看测试历史 ...")
    history = client.get_test_history()
    print(f"   历史记录数: {len(history)}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
