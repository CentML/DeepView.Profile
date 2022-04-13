import argparse
import enum
import sys

import skyline.commands.measurements
import skyline.commands.prediction_models


def main():
    parser = argparse.ArgumentParser(
        prog="skyline-evaluate",
        description="Skyline Evaluation Tool",
    )
    subparsers = parser.add_subparsers(title="Commands")
    skyline.commands.measurements.register_command(subparsers)
    skyline.commands.prediction_models.register_command(subparsers)
    args = parser.parse_args()

    if 'func' not in args:
        parser.print_help()
        sys.exit(1)

    # Run the specified command
    args.func(args)


if __name__ == '__main__':
    main()
