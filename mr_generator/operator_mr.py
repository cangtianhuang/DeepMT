class OperatorMRGenerator:
    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def generate(self, operator_ir):
        mr_functions = self.kb.get_mrs_for_operator(operator_ir.name)
        mrs = []
        for func in mr_functions:
            mr_obj = func(operator_ir.inputs)
            mrs.append(mr_obj)
        return mrs
