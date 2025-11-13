import torch


class PyTorchPlugin:
    def ir_to_code(self, ir_object, mr):
        """
        将IR和MR映射为Python可执行函数
        """

        def run():
            x = torch.tensor(ir_object.inputs[0])
            y = torch.tensor(ir_object.inputs[1])
            # 原始执行
            orig_output = torch.add(x, y)
            # MR变换后的执行
            tx, ty = mr["transform"](ir_object.inputs[0], ir_object.inputs[1])
            tx = torch.tensor(tx)
            ty = torch.tensor(ty)
            mr_output = torch.add(tx, ty)
            return orig_output, mr_output

        return run

    def execute(self, run_func):
        """
        执行传入的函数并返回结果
        """
        return run_func()
