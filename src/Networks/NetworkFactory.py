from Networks.StateOnlyNN import StateOnlyNN

class NetworkFactory:
    def __init__(self, train_config, configs=None, network_types=None):
        self.configs = configs or {"default": {"learning_rate":train_config.learning_rate}}
        self.network_types = network_types

    def create(self, name="default"):
        params = self.configs.get(name, {})
        if isinstance(self.network_types, dict):
            network_type = self.network_types.get(name, StateOnlyNN)
        else:
            network_type = self.network_types   
        return network_type(**params)