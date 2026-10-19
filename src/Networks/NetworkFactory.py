from Networks.StateOnlyNN import StateOnlyNN

class NetworkFactory:
    def __init__(self, train_config, configs=None, network_type=StateOnlyNN):
        self.configs = configs or {"default":{"learning_rate":train_config.learning_rate}}
        self.network_type = network_type

    def create(self, name="default"):
        config = self.configs.get(name, {})
        return self.network_type(**config)