class PluginsManager:
    def __init__(self):
        self.plugins = {}

    def load_plugins(self):
        # 动态加载plugins目录中的插件
        pass

    def get_plugin(self, name):
        return self.plugins[name]
