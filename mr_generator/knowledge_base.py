class KnowledgeBase:
    def __init__(self):
        # 每个算子对应可用的蜕变关系
        self.operator_mrs = {"Add": [self.commutative_mr]}

    def get_mrs_for_operator(self, operator_name):
        return self.operator_mrs.get(operator_name, [])

    def commutative_mr(self, inputs):
        """
        交换律：f(x, y) == f(y, x)
        返回一个MR对象（简单起见用 dict 表示）
        """
        x, y = inputs
        return {
            "description": "Commutative property: f(x,y) == f(y,x)",
            "transform": lambda a, b: (b, a),
            "expected": "equal",
        }
