class ModelMRGenerator:
    def __init__(self, transform_library):
        self.transforms = transform_library

    def generate(self, model_ir):
        # 静态解析结构
        layers = model_ir.get_layers()
        mrs = []
        # 根据层属性选择变换
        for t in self.transforms:
            if t.compatible_with(layers):
                mrs.append(t.to_mr(model_ir))
        return mrs
