class TaskScheduler:
    def __init__(self, ir_manager, mr_generator, plugins_manager, results_manager):
        self.ir_manager = ir_manager
        self.mr_generator = mr_generator
        self.plugins_manager = plugins_manager
        self.results_manager = results_manager

    def run_task(self, ir_object, target_framework):
        # 生成蜕变关系
        mrs = self.mr_generator.generate(ir_object)
        results = []
        for mr in mrs:
            # 加载插件
            plugin = self.plugins_manager.get_plugin(target_framework)
            framework_code = plugin.ir_to_code(ir_object, mr)
            output = plugin.execute(framework_code)
            results.append((mr, output))

        self.results_manager.compare_and_store(ir_object, results)
