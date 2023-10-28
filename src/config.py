import yaml

class Config():
    def __init__(self, yaml_cfg):
        with open(yaml_cfg, 'r') as f:
            self.cfg = yaml.safe_load(f)
        