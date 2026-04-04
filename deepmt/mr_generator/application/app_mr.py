class ApplicationMRGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client

    def generate(self, app_ir):
        # 调用LLM生成自然语言MR
        description = self.llm.ask("Generate MR for: " + app_ir.description)
        # 转换为代码
        return self.parse_description_to_code(description)
