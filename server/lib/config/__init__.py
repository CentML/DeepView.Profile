import yaml


class _Config:
    def __init__(self):
        self.Hints = None

    def initialize_hints_config(self, hints_file):
        with open(hints_file, 'r') as f:
            self.Hints = yaml.load(f)


Config = _Config()
