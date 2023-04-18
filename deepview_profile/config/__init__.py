import yaml

import deepview_profile.data


class _Config:
    def __init__(self):
        self.Hints = None

        self.warm_up = 100
        self.measure_for = 10

    def initialize_hints_config(self, hints_file):
        if hints_file is None:
            file_to_open = deepview_profile.data.get_absolute_path('hints.yml')
        else:
            file_to_open = hints_file

        with open(file_to_open, 'r') as f:
            self.Hints = yaml.load(f, Loader=yaml.Loader)

    def parse_args(self, args):
        if 'hints_file' not in args:
            args.hints_file = None
        self.initialize_hints_config(args.hints_file)

        if 'warm_up' in args and args.warm_up is not None:
            self.warm_up = args.warm_up
        if 'measure_for' in args and args.measure_for is not None:
            self.measure_for = args.measure_for


Config = _Config()
