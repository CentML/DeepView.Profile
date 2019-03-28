import yaml


class _Config:
    def __init__(self):
        self.Hints = None

        self.warm_up = 100
        self.measure_for = 10

    def initialize_hints_config(self, hints_file):
        with open(hints_file, 'r') as f:
            self.Hints = yaml.load(f, Loader=yaml.Loader)

    def parse_args(self, args):
        self.initialize_hints_config(args.hints_file)

        if args.warm_up is not None:
            self.warm_up = args.warm_up
        if args.measure_for is not None:
            self.measure_for = args.measure_for


Config = _Config()
